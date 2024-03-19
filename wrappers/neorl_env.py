import neorl
import gym
from gym import spaces
import numpy as np


class TruncatedEnv(gym.Env):
    def __init__(self, env_name) -> None:
        super().__init__()
        self._name = env_name
        self._env = neorl.make(env_name)
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        self.sim = self._env.sim
        
        self._max_episode_steps = 1000
        self.observation_space = spaces.Box(low=observation_space.low[1:], 
                                            high=observation_space.high[1:], 
                                            shape=(observation_space.shape[0] - 1, ), 
                                            dtype=observation_space.dtype)
        self.action_space = spaces.Box(low=action_space.low, high=action_space.high, shape=action_space._shape, dtype=action_space.dtype)

    def reset(self, **kwargs):
        self._episode_steps = 0
        obs = self._env.reset(**kwargs).astype(np.float32)[1:]
        return obs
    
    def step(self, action):
        self._episode_steps += 1
        obs, reward, done, info = self._env.step(action)
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        obs = obs.astype(np.float32)[1:]

        return obs, reward, done, info

    def get_dataset(self, data_type, train_num=1000):
        dataset = self._env.get_dataset(data_type=data_type, train_num=train_num)[0]
        dataset['observations'] = dataset['obs'][..., 1:]
        dataset['actions'] = dataset['action']
        dataset['rewards'] = dataset['reward'].reshape(-1)
        dataset['terminals'] = dataset['done'].reshape(-1)
        
        return dataset
        

def make_env(env_name):
    return TruncatedEnv(env_name)