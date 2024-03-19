import torch
import numpy as np
import collections

from utils.normalizer import DatasetNormalizer

ActionBatch = collections.namedtuple('ActionBatch', 'observations actions')


class ActorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, normalizer=None, device="cuda"):
        super().__init__()
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = DatasetNormalizer(dataset)

        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.device = torch.device(device)
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        observations = self.normalizer(self.observations[idx], 'observations')
        actions = self.actions[idx]
        
        return ActionBatch(observations, actions)
    
    def sample_batch(self, batch_size):
        batch_inds = np.random.choice(
            np.arange(len(self.actions)),
            size=batch_size,
            replace=True
        )
        
        observations = torch.from_numpy(self.normalizer(self.observations[batch_inds], 'observations')).float().to(self.device)
        actions = torch.from_numpy(self.actions[batch_inds]).float().to(self.device)

        return observations, actions