import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from models.utils import orthogonal_init_

LOG_STD_MIN = -5
LOG_STD_MAX = 2
epsilon = 1e-6


class DeterministicActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim=256, hidden_layer=1, max_action=1.):
        super(DeterministicActor, self).__init__()
        self.net = nn.Sequential()
        self.net.append(nn.Linear(observation_dim, hidden_dim))
        self.net.append(nn.ReLU(inplace=True))
        for _ in range(hidden_layer):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_dim, action_dim))
        
        self.apply(orthogonal_init_)
        self.max_action = max_action

    def forward(self, state):
        action = self.net(state).clip(-self.max_action, self.max_action)
        return action

    def act(self, state):
        action = self.forward(state).detach().cpu().numpy().flatten()
        return action
    
    def loss(self, state, action):
        pred = self(state)
        loss = F.mse_loss(action, pred)
        return loss


        