"""
Simple wrapper for registering metaworld enviornments
properly with gym.
"""
import gym
from gym import spaces
import metaworld
import numpy as np


class SawyerEnv(gym.Env):
    def __init__(self, env_name, select_dim=None, seed=True):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self._env = ALL_V2_ENVIRONMENTS[env_name]()
        self._env._freeze_rand_vec = False
        self._env._set_task_called = True
        self._seed = seed
        if self._seed:
            self._env.seed(0)  # Seed it at zero for now.

        if select_dim is not None:
            self.select_dim = select_dim
        else:
            self.select_dim = np.arange(self._env.observation_space.shape[0])
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        self._max_episode_steps = self._env.max_path_length
        self.observation_space = spaces.Box(low=observation_space.low[self.select_dim], 
                                            high=observation_space.high[self.select_dim], 
                                            shape=(len(self.select_dim), ), 
                                            dtype=observation_space.dtype)
        self.action_space = spaces.Box(low=action_space.low, high=action_space.high, shape=action_space._shape, dtype=action_space.dtype)

    def seed(self, seed=None):
        super().seed(seed=seed)
        if self._seed:
            self._env.seed(0)

    def evaluate_state(self, state, action):
        return self._env.evaluate_state(state, action)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, done, timeout, info = self._env.step(action)
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        obs = obs.astype(np.float32)[self.select_dim]

        return obs, reward, done, info

    def set_state(self, state):
        qpos, qvel = state[: self._env.model.nq], state[self._env.model.nq :]
        self._env.set_state(qpos, qvel)

    def reset(self, **kwargs):
        self._episode_steps = 0
        obs = self._env.reset(**kwargs)[0].astype(np.float32)[self.select_dim]
        return obs

    def render(self, mode="rgb_array", width=640, height=480):
        assert mode == "rgb_array", "Only RGB array is supported"
        # stack multiple views
        view_1 = self._env.render(offscreen=True, camera_name="corner", resolution=(width, height))
        view_2 = self._env.render(offscreen=True, camera_name="topview", resolution=(width, height))
        return np.concatenate((view_1, view_2), axis=0)

    def __getattr__(self, name):
        return getattr(self._env, name)



