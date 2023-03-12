from typing import Tuple

import gym
from gym.core import ObsType, ActType
from gym.wrappers import TimeLimit, FrameStack


class WalkerEnvWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, dict = self.env.reset(**kwargs)
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, truncated, info = self.env.step(action)
        obs = self.env.render()
        return obs, rew, done, info


def create_walker_env(hardcore=True, max_step=2000, num_stack=4):
    env = WalkerEnvWrapper(
        FrameStack(
            TimeLimit(gym.make('BipedalWalker-v3', render_mode='rgb_array', hardcore=hardcore),
                      max_episode_steps=max_step), num_stack=num_stack))
    return env
