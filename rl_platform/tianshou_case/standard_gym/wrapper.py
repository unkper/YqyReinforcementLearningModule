from typing import Tuple

import gym
from gym.core import ObsType, ActType
from gym.wrappers import TimeLimit, FrameStack

from rl_platform.tianshou_case.third_party.single_curiosity_env_wrapper import resize_observation

target_image_shape = [80, 120, 3]


class WalkerEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = target_image_shape

    def reset(self, **kwargs):
        obs, dict = self.env.reset(**kwargs)
        obs = self.env.render()
        obs = resize_observation(obs, target_image_shape)
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, truncated, info = self.env.step(action)
        obs = self.env.render()
        obs = resize_observation(obs, target_image_shape)
        return obs, rew, done, info


def create_walker_env(hardcore=False, max_step=2000, num_stack=4):
    env = WalkerEnvWrapper(
        FrameStack(
            TimeLimit(gym.make('BipedalWalker-v3', render_mode='rgb_array', hardcore=hardcore),
                      max_episode_steps=max_step), num_stack=num_stack))
    return env
