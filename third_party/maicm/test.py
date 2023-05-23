import multiprocessing
import random
import time

import torch
import numpy as np

from ped_env.interfaces.mn_transfer_wrapper import create_ped_env_mn
from ped_env.interfaces.maicm_interface import create_ped_env
from third_party.maicm.main import make_parallel_env
from third_party.maicm.utils.env_wrappers import SubprocVecEnv
from utils.misc import apply_to_all_elements
from algorithms.sac import SAC

agent_num = 4

def make_parallel_test_env(config, seed):
    global agent_num
    lock = multiprocessing.Lock()

    def get_env_fn(rank):
        def init_env():
            if config.env_type == 'pedsmove':
                env = create_ped_env_mn(map=config.map_ind,
                                     leader_num=config.num_agents,
                                     group_size=config.group_size,
                                     maxStep=config.max_episode_length,
                                     frame_skip=config.frame_skip,
                                     seed=(seed * 1000),
                                     use_adv_net=config.use_adv_encoder,
                                     use_concat_obs=config.use_concat_obs,
                                     agent_num=agent_num)
            else:  # vizdoom
                raise Exception("该修改代码不支持Vizdoom环境!")
            return env

        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in
                          range(config.n_rollout_threads)])

def test(config, load_file, episode=2):
    run_num = random.randint(1, 9999)

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_test_env(config, run_num)
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
            #time.sleep(0.02)
    env.close(force=(config.env_type == 'vizdoom'))

if __name__ == "__main__":
    from third_party.maicm.params.ped import params1 as ped_p
    config = ped_p.Params("map_12", 16, 1)
    #config.args = ped_p.debug_mode(config.args)
    config.args.n_rollout_threads = 1

    path = r"D:\projects\python\PedestrainSimulationModule\third_party\maicm\models\pedsmove\final_test\map12\2023_04_30_22_58_36exp_test\run1\model.pt"

    test(config.args, path)