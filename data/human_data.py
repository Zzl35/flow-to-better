import torch
import numpy as np
import collections

from .scripted_data import ScriptedDataset


Batch = collections.namedtuple('Batch', 'trajectories conditions timesteps')


class HumanDataset(ScriptedDataset):
    def __init__(self, task, dataset, human_label_dataset, normalizer, max_len=100, pref_num=500, device="cuda"):
        super().__init__(task, dataset, normalizer, max_len, pref_num, device)
        human_label_dataset = {k: v for k, v in human_label_dataset.items()}

        self.human_obss_1 = torch.as_tensor(human_label_dataset["observations"], dtype=torch.float32, device=self.device)[:pref_num]
        self.human_obss_2 = torch.as_tensor(human_label_dataset["observations_2"], dtype=torch.float32, device=self.device)[:pref_num]
        self.human_actions_1 = torch.as_tensor(human_label_dataset["actions"], dtype=torch.float32, device=self.device)[:pref_num]
        self.human_actions_2 = torch.as_tensor(human_label_dataset["actions_2"], dtype=torch.float32, device=self.device)[:pref_num]
        self.timesteps_1 = torch.as_tensor(human_label_dataset["fixed_timestep_1"], dtype=torch.long, device=self.device)[:pref_num]
        self.timesteps_2 = torch.as_tensor(human_label_dataset["fixed_timestep_2"], dtype=torch.long, device=self.device)[:pref_num]
        self.preferences = torch.as_tensor(human_label_dataset["labels"], dtype=torch.float32, device=self.device)[:pref_num]
        
        num_labels = self.human_obss_1.shape[0]
        num_label_0 = (self.preferences[:, 0]==1).sum()
        num_label_1 = (self.preferences[:, 1]==1).sum()
        num_label_2 = (self.preferences[:, 0]==0.5).sum()
        print(f'num labels: {num_labels}')
        print(f'label_0 ratio: {num_label_0/num_labels:.4f}')
        print(f'label_1 ratio: {num_label_1/num_labels:.4f}')
        print(f'label_2 ratio: {num_label_2/num_labels:.4f}')
    
    def sample_prefs(self, batch_size=256):
        batch_inds = torch.randint(0, self.human_obss_1.size(0), size=(batch_size,))
        obss_1, obss_2 = self.human_obss_1[batch_inds], self.human_obss_2[batch_inds]
        obss_1, obss_2 = self.normalizer(obss_1, 'observations'), self.normalizer(obss_2, 'observations')
        actions_1, actions_2 = self.human_actions_1[batch_inds], self.human_actions_2[batch_inds]
        actions_1, actions_2 = self.normalizer(actions_1, 'actions'), self.normalizer(actions_2, 'actions')
        timesteps_1, timesteps_2 = self.timesteps_1[batch_inds], self.timesteps_2[batch_inds]
        prefs = self.preferences[batch_inds]

        masks_1, masks_2 = timesteps_1.clip(0, 1), timesteps_2.clip(0, 1)
        masks_1[..., 0] = 1
        masks_2[..., 0] = 1

        return (obss_1[:, :self.max_len], obss_2[:, :self.max_len]), \
            (actions_1[:, :self.max_len], actions_2[:, :self.max_len]), \
            (timesteps_1[:, :self.max_len], timesteps_2[:, :self.max_len]), \
            (masks_1[:, :self.max_len], masks_2[:, :self.max_len]), \
            prefs
    