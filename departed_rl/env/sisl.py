import time

import numpy as np
from numpy import inf
from gym.spaces import Box

from pettingzoo.sisl import waterworld_v3
from pettingzoo.sisl import multiwalker_v7
from pettingzoo.utils import random_demo
from gym import Env
from numpy import float32

class MultiWalker(Env):
    def __init__(self):
        super(MultiWalker, self).__init__()
        self.wrappedEnv = multiwalker_v7.parallel_env()
        self.agent_count = 3
        self.action_lim = 1.0
        self.observation_space = [Box(-inf, inf, (31,), float32),
                                  Box(-inf, inf, (31,), float32),
                                  Box(-inf, inf, (31,), float32)
                                  ]
        self.action_space = [Box(-self.action_lim, self.action_lim, (4,), float32),
                             Box(-self.action_lim, self.action_lim, (4,), float32),
                             Box(-self.action_lim, self.action_lim, (4,), float32),
                             ]

    def step(self, action):
        import copy
        _action = copy.copy(action)
        actions = {agent: _action[idx] for idx,agent in enumerate(self.wrappedEnv.agents)}
        obs ,reward, is_done, info = self.wrappedEnv.step(actions)
        obs = list(obs.values())
        reward = list(reward.values())
        is_done = list(is_done.values())
        return obs, reward, is_done, "MultiWalker"

    def reset(self):
        obs = self.wrappedEnv.reset()
        obs = list(obs.values())
        return obs

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

class WaterWorld(Env):
    def __init__(self):
        self.max_accel = 0.01
        self.wrappedEnv = waterworld_v3.parallel_env()
        self.agent_count = 2
        self.action_lim = self.max_accel
        self.observation_space = [Box(-1.4142135381698608,2.8284270763397217, (242,), float32),
                                  Box(-1.4142135381698608,2.8284270763397217, (242,), float32),
                                  ]
        self.action_space = [Box(-self.max_accel, self.max_accel, (2,), float32),
                             Box(-self.max_accel, self.max_accel, (2,), float32),
                             ]

    def step(self, action):
        import copy
        _action = copy.copy(action)
        _action = _action.detach().cpu().numpy()
        obs ,reward, is_done = self.wrappedEnv.step(_action)
        return obs, reward, is_done, "WaterWorld"

    def reset(self):
        obs = self.wrappedEnv.reset()
        return obs

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

if __name__ == '__main__':
    # ped_env = multiwalker_v7.parallel_env()
    # print(ped_env.observation_spaces)
    # print(ped_env.action_spaces)
    env = MultiWalker()
    print(env.reset())
    for i in range(75):
        obs, rewards, is_done, _ = env.step()
