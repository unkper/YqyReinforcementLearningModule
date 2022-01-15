import random

from random import sample

import numpy as np
import torch
from gym import Env
from torch import nn

from rl.utils.functions import get_dict

def greedy_policy(A, s, Q, epsilon = None):
    """在给定一个状态下，从行为空间A中选择一个行为a，使得Q(s,a) = max(Q(s,))
    考虑到多个行为价值相同的情况返回一个随机行为
    """
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    return random.choice(a_max_q)

def uniform_random_policy(A, s = None, Q = None):
    '''均一随机策略
    '''
    if type(A) not in [list, set]:
        return A.sample()
    return sample(A,k=1)[0]

def epsilon_greedy_policy(A, s, Q, epsilon = 0.05):
    rand_value = random.random()
    if rand_value < epsilon:
        if type(A) not in [list,set]:
            return A.sample()
        return sample(A,k=1)[0]
    else:
        return greedy_policy(A, s, Q)

def deep_epsilon_greedy_policy(s, epsilon, env:Env, behavior_Q:nn.Module):
    action = behavior_Q(s)  # 经过神经网络输出一个[1,5]向量，里面代表前，后，左，右，不动的值，之后选取其中最大的输出
    if isinstance(action, torch.Tensor):action = action.detach().cpu().numpy()
    rand_value = random.random()
    if epsilon is not None and rand_value < epsilon:
        return env.action_space.sample()  # 有ε概率随机选取动作
    else:
        return int(np.argmax(action))