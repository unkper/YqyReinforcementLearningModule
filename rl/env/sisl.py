import time

import numpy as np
from gym.spaces import Box

from pettingzoo.sisl import waterworld_v3
from pettingzoo.utils import random_demo
from gym import Env
from numpy import float32

class WaterWorld(Env):
    def __init__(self):
        self.max_accel = 0.01
        self.wrappedEnv = waterworld_v3.env(n_pursuers=2, n_coop=2, sensor_range=0.2,
                                            food_reward=10, poison_reward=-1, encounter_reward=0.01,
                                            pursuer_max_accel=self.max_accel)
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
        obs ,reward, is_done = [], [], []
        for i in range(self.agent_count):
            o, r, _isDone, _ = self.wrappedEnv.last()
            obs.append(o)
            reward.append(r)
            is_done.append(_isDone)
            if not _isDone:
                self.wrappedEnv.step(_action[i] * self.max_accel)
            else:
                self.wrappedEnv.step(None)

        return obs, reward, is_done, "WaterWorld"

    def reset(self):
        self.isDone = [False for _ in range(self.agent_count)]
        self.wrappedEnv.reset()
        obs = []
        for agent in self.wrappedEnv.agents:
            obs.append(self.wrappedEnv.observe(agent))
        return obs

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

if __name__ == '__main__':
    import torch
    env = WaterWorld()
    obs = env.reset()
    is_done = [False]
    while not is_done[0]:
        obs,reward,is_done,_ = env.step(torch.rand((5,2)))
        print(obs,reward,is_done)

