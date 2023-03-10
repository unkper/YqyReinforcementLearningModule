import gym
import numpy as np

from ped_env.envs import PedsMoveEnv

from ped_env.utils.maps import map_simple
from departed_rl.utils.functions import onehot_from_int


class SinglePedEnv(gym.Env):
    def __init__(self):
        self.wrapper_env = PedsMoveEnv(terrain=map_simple, person_num=1, group_size=(1, 1))
        self.observation_space = self.wrapper_env.observation_space[0]
        self.action_space = self.wrapper_env.action_space[0]

    def step(self, action):
        action = [onehot_from_int(action, 9)]
        o, r, d, i = self.wrapper_env.step(action)
        return np.array(o[0]), np.array(r[0]), np.array(d[0]), i

    def reset(self):
        obs = self.wrapper_env.reset()
        return np.array(obs[0])
