from gym_derk.envs import DerkEnv

import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts

task = 'DerkEnv'

def derk_env():
    from gym_derk.envs import DerkEnv
    from gym_derk import ObservationKeys
    import numpy as np
    import gym
    import math
    import os.path

    env = DerkEnv()
    env.mode = 'train'

    print(env.observation_space)
    print(env.action_space)

    env.reset()
    is_done = False
    while not is_done:
        actions = []
        for i in range(len(env.action_space)):
            actions.append(env.action_space[i].sample())
        env.step(actions)



derk_env()