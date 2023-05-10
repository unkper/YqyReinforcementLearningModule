import logging
import random

import torch
import os
import multiprocessing
import numpy as np
import tqdm

from pathlib import Path
from collections import deque
from tensorboardX import SummaryWriter

from third_party.maicm.main import make_parallel_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv
from utils.misc import apply_to_all_elements, timeout, RunningMeanStd
from algorithms.sac import SAC
from envs.magw.multiagent_env import GridWorld, VectObsEnv
from ped_env.utils.maicm_interface import create_ped_env


def test(config, load_file, episode=2):
    run_num = random.randint(1, 9999)

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config, run_num)
    active_envs = np.ones(config.n_rollout_threads)  # binary indicator of whether env is active

    model = SAC.init_from_save(load_file, load_critic=True, load_ir=True)
    for e in range(episode):

        steps_since_update = 0
        state, obs = env.reset()
        dones = [False]
        while not all(dones):
            model.prep_rollouts(device='cuda' if config.gpu_rollout else 'cpu')
            # convert to torch tensor
            torch_obs = apply_to_all_elements(obs, lambda x: torch.tensor(x, dtype=torch.float32,
                                                                          device='cuda' if config.gpu_rollout else 'cpu'))
            # get actions as torch tensors
            torch_agent_actions = model.step(torch_obs, explore=True) # 符合CTDE架构
            # convert actions to numpy arrays
            agent_actions = apply_to_all_elements(torch_agent_actions, lambda x: x.cpu().data.numpy())
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(int(active_envs.sum()))]

            next_state, next_obs, rewards, dones, infos = env.step(actions, env_mask=active_envs)

            obs = next_obs
            env.render()
            steps_since_update += int(active_envs.sum())
    env.close(force=(config.env_type == 'vizdoom'))

if __name__ == "__main__":
    from third_party.maicm.params.ped import params1 as ped_p
    config = ped_p.Params("map_10", 4, 1)
    #config.args = ped_p.debug_mode(config.args)
    config.args.n_rollout_threads = 1

    path = r"D:\projects\python\PedestrainSimulationModule\third_party\maicm\models\pedsmove\final_test\map10\2023_05_02_16_06_07exp_test\run1\model.pt"

    test(config.args, path)