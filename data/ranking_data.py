import numpy as np
import collections
import torch
import copy
from sklearn.cluster import KMeans

from .human_data import HumanDataset

Batch = collections.namedtuple('Batch', 'observations actions masks next_observations')


class BlockRankingDataset(HumanDataset):
    def __init__(self, task, dataset, normalizer, human_label_dataset, max_len=1000, pref_num=500, device="cuda"):
        super().__init__(task, dataset, human_label_dataset, normalizer, max_len, pref_num, device)
        
        self.observations, self.actions, self.next_observations = [], [], []
        for traj in self.trajs:
            self.observations.extend(traj['observations'][:-1])
            self.actions.extend(traj['actions'][:-1])
            self.next_observations.extend(traj['observations'][1:])
        self.observations = self.normalizer(np.array(self.observations), 'observations')
        self.actions = self.normalizer(np.array(self.actions), 'actions')
        self.next_observations = self.normalizer(np.array(self.next_observations), 'observations')

        self.observation_dim = self.observations.shape[-1]
        
        # self.block_ranking(improve_step)

    def set_returns(self, pref_model, discount=1., batch_size=32):    
        self.returns_ = copy.deepcopy(self.returns)
        start = 0
        while start < len(self.trajs):
            end = min(start + batch_size, len(self.trajs))
            observations, actions, returns, timesteps, masks = self.get_batch(np.arange(start, end))
            self.returns[start: end] = pref_model._predict_traj_return(observations, actions, timesteps, masks, discount=discount)
            start += batch_size

        self.min_return = self.returns.min()
        self.max_return = self.returns.max()
        self.returns = (self.returns - self.returns.min()) / (self.returns.max() - self.returns.min())
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        observations = self.observations[idx]
        actions = self.actions[idx]
        next_observations = self.next_observations[idx]
        
        return Batch(observations, actions, next_observations)
    
    def block_ranking(self, improve_step=10):
        self.ranking_indices = sorted(range(len(self.returns)), key=lambda k: self.returns[k])
        self.ranking_returns = self.returns[self.ranking_indices]
        
        self.improve_step = improve_step

        km_cluster = KMeans(n_clusters=improve_step, n_init=40, init='k-means++')
        labels = km_cluster.fit_predict(self.ranking_returns.reshape(-1, 1))

        block_indices = [[] for _ in range(self.improve_step)]
        block_idx = 0
        count = 0
        for idx, value in enumerate(self.ranking_returns):
            if idx > 0 and labels[idx] != labels[idx - 1]:
                block_idx += 1
            block_indices[block_idx].append(self.ranking_indices[idx])
            count += 1
        self.block_indices = [np.array(indices) for indices in block_indices]
        self.block_size = np.array([len(x) for x in self.block_indices])
        
        for i in range(improve_step):
            indices = self.block_indices[i]
            min_return = self.returns[indices[0]]
            max_return = self.returns[indices[-1]]
            size = self.block_size[i]
            print('Block %d: min=%.4f  max=%.4f  size=%d'%(i, min_return, max_return, size))
    
    def get_block_id(self, score):
        for i in range(self.improve_step):
            indices = self.block_indices[i]
            min_return = self.returns[indices[0]]
            max_return = self.returns[indices[-1]]
            if min_return <= score <= max_return:
                return i
        return self.improve_step + 1

    def sample_init(self, num):
        batch_inds = torch.randint(0, self.init_obs.shape[0], size=(num,))
        init_obs = self.init_obs[batch_inds]
        init_obs = torch.from_numpy(self.normalizer(init_obs, 'observations')).float().to(self.device).unsqueeze(dim=1)
        return init_obs
    
    def get_block_traj(self, idx):
        indices = self.block_indices[idx]
        observations = []
        for idx in indices:
            obs, _, _, _ = self.get_traj(idx)
            observations.append(obs)
        return np.array(observations)
    
    def get_top_traj(self, sample_num):
        if sample_num > len(self.ranking_indices):
            sample_num = len(self.ranking_indices)
        sample_inds = self.ranking_indices[-sample_num:]
        
        sample_obs = []
        sample_act = []
        for idx in sample_inds:
            observations, actions, _, _ = self.get_traj(idx)
            sample_obs.append(observations)
            sample_act.append(actions)
        sample_obs = np.array(sample_obs)
        sample_act = np.array(sample_act)
        sample_traj = np.concatenate([sample_obs, sample_act], axis=-1)

        condition_traj = torch.from_numpy(sample_traj).float().to(self.device)
        init_obs = condition_traj[:, 0: 1, :self.observation_dim]
            
        return condition_traj, init_obs
    
    def sample_batch(self, batch_size, mode="train"):
        if mode == "train":
            sample_size_per_block = np.ceil(batch_size / (self.improve_step - 1)).astype(int)
            
            sample_inds = []
            for indices in self.block_indices: 
                batch_inds = np.random.choice(
                    np.arange(len(indices)),
                    size=sample_size_per_block,
                    replace=True
                )
                sample_inds.extend(indices[batch_inds])
            sample_inds = sample_inds
        else:
            batch_inds = np.random.choice(
                np.arange(len(self.block_indices[-1])),
                size=batch_size,
                replace=True
            )
            sample_inds = self.block_indices[-1][batch_inds]
            
        sample_obs = []
        sample_act = []
        sample_mask = []
        for idx in sample_inds:
            observations, actions, _, masks = self.get_traj(idx)
            sample_obs.append(observations)
            sample_act.append(actions)
            sample_mask.append(masks)
        sample_obs = np.array(sample_obs)
        sample_act = np.array(sample_act)
        sample_mask = np.array(sample_mask)
        sample_traj = np.concatenate([sample_obs, sample_act], axis=-1)
        
        if mode == "train":
            condition_traj = torch.from_numpy(sample_traj[:-sample_size_per_block]).float().to(self.device)
            target_traj = torch.from_numpy(sample_traj[sample_size_per_block:]).float().to(self.device)
            masks = torch.from_numpy(sample_mask[sample_size_per_block:]).float().to(self.device).unsqueeze(dim=-1)
        else:
            condition_traj = torch.from_numpy(sample_traj).float().to(self.device)
            target_traj = torch.from_numpy(sample_traj).float().to(self.device)
            masks = torch.from_numpy(sample_mask).float().to(self.device).unsqueeze(dim=-1)
        init_obs = target_traj[:, 0: 1, :self.observation_dim]

        return condition_traj, target_traj, init_obs, masks
    
    
    
