import torch
import numpy as np
import collections


def get_discount_returns(rewards, discount=1):
    returns = 0
    scale = 1
    for r in rewards:
        returns += scale * r
        scale *= discount
    return returns
    
    
class ScriptedDataset(torch.utils.data.Dataset):
    def __init__(self, task, dataset, normalizer, max_len=100, pref_num=500, device="cuda"):
        super().__init__()

        self.normalizer = normalizer
        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.max_len = max_len
        self.device = torch.device(device)

        data_ = collections.defaultdict(list)
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        self.trajs = []
        self.filter_trajs = []
        dataset = {k: v for k, v in dataset.items()}
        for i in range(dataset["rewards"].shape[0]):
            for k in ['observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            episode_step += 1
            terminal = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == max_len)
            if final_timestep or terminal:
                episode_data = {}
                # if terminal: data_['rewards'][-1] = -100
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                if final_timestep and episode_step < max_len:
                    pass
                else:
                    self.trajs.append(episode_data)
                episode_step = 0
                data_ = collections.defaultdict(list)

        self.pref_trajs = np.random.randint(0, len(self.trajs), 2 * pref_num).reshape(pref_num, 2)
        self.init_obs = np.array([traj["observations"][0] for traj in self.trajs])
    
        self.returns = np.array([get_discount_returns(t['rewards']) for t in self.trajs])
        num_samples = np.sum([t['rewards'].shape[0] for t in self.trajs])
        
        print(f'Number of samples collected: {num_samples}')
        print(f'Num trajectories: {len(self.trajs)}')
        print(f'Trajectory returns: mean = {np.mean(self.returns)}, std = {np.std(self.returns)}, max = {np.max(self.returns)}, min = {np.min(self.returns)}')

    def get_traj(self, idx):
        traj = self.trajs[idx]
        observations = traj['observations']
        actions = traj['actions']
        tlen = min(len(observations), self.max_len)

        observations = np.concatenate([observations, np.zeros((self.max_len-tlen, self.obs_dim))], 0).astype(np.float32)
        actions = np.concatenate([actions, np.zeros((self.max_len-tlen, self.action_dim))], 0).astype(np.float32)
        timesteps = np.concatenate([np.arange(0, tlen), np.zeros(self.max_len - tlen)], axis=0)
        masks = np.ones_like(timesteps)
        masks[tlen:] *= 0

        observations = self.normalizer(observations, 'observations')
        actions = self.normalizer(actions, 'actions')
        
        observations = observations[: self.max_len]
        actions = actions[: self.max_len]
        timesteps = timesteps[: self.max_len]
        masks = masks[: self.max_len]
        
        return observations, actions, timesteps, masks
    
    def get_batch(self, batch_inds):
        obss_, actions_, rtgs_, timesteps_, masks_ = [], [], [], [], []

        for idx in batch_inds:
            obss, actions, timesteps, mask = self.get_traj(idx)
            rtg = self.returns[idx]

            obss_.append(obss)
            actions_.append(actions)
            rtgs_.append(rtg)
            timesteps_.append(timesteps)
            masks_.append(mask)
        
        obss = torch.from_numpy(np.stack(obss_, 0)).float().to(self.device)
        actions = torch.from_numpy(np.stack(actions_, 0)).float().to(self.device)
        rtgs = torch.from_numpy(np.array(rtgs_)).float().to(self.device)
        timesteps = torch.from_numpy(np.stack(timesteps_)).long().to(self.device)
        masks = torch.from_numpy(np.stack(masks_, 0)).long().to(self.device)

        return obss, actions, rtgs, timesteps, masks

    def sample_batch(self, batch_size):
        batch_inds = np.random.choice(
            np.arange(len(self.trajs)),
            size=batch_size,
            replace=True
        )

        observations = []
        for idx in batch_inds:
            obs, _, _, _ = self.get_traj(idx)
            observations.append(obs)
        observations = np.array(observations)

        condition_obs = torch.from_numpy(observations).float().to(self.device)
        target_obs = torch.from_numpy(observations).float().to(self.device)
        init_obs = target_obs[:, 0: 1]
        returns = torch.from_numpy(self.returns[batch_inds].reshape(-1, 1)).float().to(self.device)

        return condition_obs, init_obs, target_obs, returns