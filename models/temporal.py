import math
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.distributions import Bernoulli
import torch.nn.functional as F


from models.utils import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock
)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class QKVAttention(nn.Module):
    def __init__(self, heads = 4) -> None:
        super().__init__()
        
        self.heads = heads
        
    def forward(self, x, encoder_kv):
        q, k, v = rearrange(x, 'b (qkv heads c) h -> qkv (b heads) c h', heads=self.heads, qkv=3)
        encoder_k, encoder_v = rearrange(encoder_kv, 'b (kv heads c) h -> kv (b heads) c h', heads=self.heads, kv=2)
        k = torch.cat([encoder_k, k], dim=-1)
        v = torch.cat([encoder_v, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(q.shape[-1]))
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = torch.einsum("bts,bcs->bct", weight, v)
        out = rearrange(out, '(b heads) c h -> b (heads c) h', heads=self.heads)
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, encoder_channels, heads=4, kernel_size=5, mish=True) -> None:
        super().__init__()
        assert out_channels % heads == 0
        
        self.blocks = nn.ModuleList([
            nn.Conv1d(inp_channels, out_channels * 3, 1),
            QKVAttention(heads),
        ])
        
        self.encoder_kv = nn.Conv1d(encoder_channels, out_channels * 2, 1)
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()
    
    def forward(self, x, t, cond):
        encoder_kv = self.encoder_kv(cond)
        out = self.blocks[0](x)
        out = self.blocks[1](out, encoder_kv)
        
        return out + self.residual_conv(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()
            
        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, cond):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class TrajCondUnet(nn.Module):
    def __init__(self, 
                 horizon, 
                 transition_dim, 
                 hidden_dim=128, 
                 dim_mults=(1, 2, 4, 8), 
                 condition_dropout=0.25, 
                 kernel_size=5):
        super().__init__()

        dims = [transition_dim, *map(lambda m: hidden_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        mish = True
        act_fn = nn.Mish()

        self.time_dim = hidden_dim
        self.embed_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            act_fn,
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.traj_conv = nn.Sequential(
            Conv1dBlock(transition_dim, self.embed_dim, kernel_size=kernel_size, mish=mish),
            Conv1dBlock(self.embed_dim, self.embed_dim, kernel_size=kernel_size, mish=mish)
        )
        self.condition_dropout = condition_dropout
        self.mask_dist = Bernoulli(probs=1-self.condition_dropout)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=self.embed_dim, kernel_size=kernel_size, mish=mish),
                ResidualAttentionBlock(dim_out, dim_out, encoder_channels=self.embed_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=self.embed_dim, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=self.embed_dim, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualAttentionBlock(mid_dim, mid_dim, self.embed_dim)
        self.mid_block3 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=self.embed_dim, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=self.embed_dim, kernel_size=kernel_size, mish=mish),
                ResidualAttentionBlock(dim_in, dim_in, encoder_channels=self.embed_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=self.embed_dim, kernel_size=kernel_size, mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(hidden_dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, use_dropout=True, force_dropout=False):
        x = einops.rearrange(x, 'b h t -> b t h')
        cond = einops.rearrange(cond, 'b h t -> b t h')

        t = self.time_mlp(time)
        c = self.traj_conv(cond)
        
        if use_dropout:
            mask = self.mask_dist.sample(sample_shape=(c.size(0), 1)).to(c.device)
            c = torch.einsum('ij,ikh->ikh', mask, c)
        if force_dropout:
            c = 0 * c

        h = []

        for resnet1, attention1, resnet2, downsample in self.downs:
            x = resnet1(x, t, c)
            x = attention1(x, t, c)
            x = resnet2(x, t, c)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_block2(x, t, c)
        x = self.mid_block3(x, t, c)

        for resnet1, attention1, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t, c)
            x = attention1(x, t, c)
            x = resnet2(x, t, c)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        
        return x
