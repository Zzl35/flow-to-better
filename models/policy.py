from collections import namedtuple
import numpy as np
import torch
import einops

from utils.helpers import apply_dict, to_torch


Trajectories = namedtuple('Trajectories', 'actions observations values')


class DiffusionPolicy:

    def __init__(self, guide, diffusion_model, normalizer, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        # self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    @ torch.no_grad()
    def __call__(self, conditions, t, batch_size=1, verbose=True):
        # conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, verbose=verbose, **self.sample_kwargs)
        trajectories = samples.trajectories.detach().cpu().numpy()

        ## extract action [ batch_size x horizon x transition_dim ]
        actions, obss = trajectories[:, :, :self.action_dim], trajectories[..., self.action_dim:]
        # log_likelihoods = samples.log_likelihoods.flatten().cpu().numpy()
        # filter = log_likelihoods > np.mean(log_likelihoods)
        # actions, obss = actions[filter], obss[filter]
        # print(len(actions))

        ## get preferences
        end = min(t+self.diffusion_model.horizon, 1000)
        timesteps_ = np.arange(t, end)
        timesteps, masks = np.zeros(obss.shape[1]), np.ones((obss.shape[0], obss.shape[1]))
        timesteps[:len(timesteps_)] = timesteps_
        masks[:, len(timesteps_):] = 0
        obss_tensor = to_torch(obss, device=self.device)
        actions_tensor = to_torch(actions, device=self.device)
        timesteps_tensor = to_torch(timesteps, device=self.device)
        masks_tensor = to_torch(masks, device=self.device)
        preferences = self.guide(obss_tensor, actions_tensor, timesteps_tensor.long(), masks_tensor.long())

        actions = self.normalizer.unnormalize(actions, 'actions')
        selected_idx = preferences.flatten().argmax()
        # print(f'max value: {preferences.max():.4f}, min value: {preferences.min():.4f}')

        ## extract first action
        # selected_idx = 0
        action = actions[selected_idx, 0]

        obss = self.normalizer.unnormalize(obss, 'observations')

        trajectories = Trajectories(actions, obss, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
