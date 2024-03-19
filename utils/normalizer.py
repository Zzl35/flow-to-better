import numpy as np
import torch


class GaussianNormalizer:
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, data):
        self.data = data.astype(np.float32)
        self.means = self.data.mean(axis=0)
        # self.stds = self.data.std(axis=0) + 1e-3
        self.stds = np.array([self.data[..., i].std() for i in range(data.shape[-1])]) + 1e-5

    def normalize(self, x):                 
        if torch.is_tensor(x):
            device = x.device
            x = (x.cpu().numpy() - self.means) / self.stds
            return torch.as_tensor(x, device=device)
        else:
            return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means


class MinMaxNormalizer:
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, data):
        self.data = data.astype(np.float32)

        self.maxs = self.data.max(axis=0)
        self.mins = self.data.min(axis=0)

    def normalize(self, x):
        if torch.is_tensor(x):
            device = x.device
            x = (x.cpu().numpy() - self.mins) / (self.maxs - self.mins)
            return torch.as_tensor(x, device=device)
        else:
            return (x - self.mins) / (self.maxs - self.mins)

    def unnormalize(self, x):
        return x *  (self.maxs - self.mins) + self.mins


class DatasetNormalizer:
    def __init__(self, dataset, mode="gaussian"):

        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]

        self.normalizers = {}
        if mode == "gaussian":
            self.normalizers['observations'] = GaussianNormalizer(dataset['observations'])
            self.normalizers['actions'] = GaussianNormalizer(dataset['actions'])
        elif mode == "minmax":
            self.normalizers['observations'] = MinMaxNormalizer(dataset['observations'])
            self.normalizers['actions'] = MinMaxNormalizer(dataset['actions'])
        else:
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)
    