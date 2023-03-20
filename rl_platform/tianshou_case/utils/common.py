from typing import Optional, Tuple, Callable, List

import numpy as np
import pettingzoo as pet
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from tianshou.env import PettingZooEnv
from tianshou.policy import BasePolicy, MultiAgentPolicyManager


def _get_agents(
        env,
        agent_learn: Optional[BasePolicy] = None,
        agent_count: int = 1,
        optim: Optional[torch.optim.Optimizer] = None,
        file_path=None,
        get_policy: Callable = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    if agent_learn is None:
        # model
        agent_learn, optim = get_policy(env, optim)

    agents = [agent_learn for _ in range(agent_count)]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def save_video(abs_file_path, obs_arr):
    fig, ax = plt.subplots()

    now_obs_idx = 0

    def update(frame):
        nonlocal now_obs_idx
        ax.clear()

        ax.imshow(obs_arr[now_obs_idx][0])
        # ax.imshow(np.transpose(obs_arr[now_obs_idx], (1, 2, 0)))
        now_obs_idx += 1
        now_obs_idx %= len(obs_arr)
        # 隐藏坐标轴
        ax.axis('off')

    ani = FuncAnimation(fig, update)
    ani.save(abs_file_path, writer='ffmpeg')


def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


import json
import importlib.util


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.__class__.__name__
