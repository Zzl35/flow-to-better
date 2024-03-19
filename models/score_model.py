import numpy as np
import torch
import torch.nn as nn


class ScoreModel(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        survival_reward=False,
        hidden_dim=256, 
        hidden_layer=1, 
        device=torch.device("cuda")
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        
        self.net = nn.Sequential()
        self.net.append(nn.Linear(observation_dim + action_dim, hidden_dim))
        self.net.append(nn.ReLU(inplace=True))
        for _ in range(hidden_layer):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_dim, 1))

        self.survival_reward = survival_reward
        
    def forward(self, observations, actions, timesteps, masks):
        x = torch.cat([observations, actions], dim=-1)
        x = self.net(x)
        x = x.mean(dim=1)
        
        return x
    
    def _predict_traj_return(self, observations, actions, timesteps, masks, discount=1):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).to(self.device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        if isinstance(timesteps, np.ndarray):
            timesteps = torch.from_numpy(timesteps).long().to(self.device)
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks).long().to(self.device)

        x = torch.cat([observations, actions], dim=-1)
        x = self.net(x)
        if self.survival_reward:
            x = torch.sigmoid(x)
        for i in range(x.shape[1]):
            x[:, i] *= discount ** i
        x = (x * masks.unsqueeze(dim=-1)).sum(dim=1).squeeze(dim=-1).detach().cpu().numpy()
        
        return x


    def predict_traj_return(self, trajs, tlens, min_r=None, max_r=None, discount=1.):
        prefs = []
        
        for tlen, traj in zip(tlens, trajs):
            observations = traj[:, :self.observation_dim].reshape(1, -1, self.observation_dim)
            actions = traj[:, self.observation_dim:].reshape(1, -1, self.action_dim)
            timesteps = torch.range(0, len(traj) - 1).long().reshape(1, -1).to(self.device)
            masks = torch.ones_like(timesteps).long().reshape(1, -1).to(self.device)
            
            timesteps[:, tlen:] *= 0
            masks[:, tlen:] *= 0
            
            prefs.append(self._predict_traj_return(observations, actions, timesteps, masks)[0])
        
        returns = np.array(prefs)
        if min_r is not None:
            returns = (returns - min_r) / (max_r - min_r)
        return returns



        