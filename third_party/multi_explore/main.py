import pickle
from typing import List

import dill
import pandas as pd
import logging

import torch
import os
import multiprocessing
import numpy as np
import tqdm

from pathlib import Path
from collections import deque, defaultdict
from tensorboardX import SummaryWriter

from ped_env.utils.misc import strf_now_time
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv
from utils.misc import apply_to_all_elements, timeout, RunningMeanStd, save_params
from algorithms.sac import SAC

from envs.magw.multiagent_env import GridWorld, VectObsEnv
from ped_env.utils.maicm_interface import create_ped_env

AGENT_CMAPS = ['Reds', 'Blues', 'Greens', 'Wistia']


def get_count_based_novelties(env, state_inds, device='cpu'):
    env_visit_counts = env.get_visit_counts()

    # samp_visit_counts[i,j,k] is # of times agent j has visited the state that agent k occupies at time i
    samp_visit_counts = np.concatenate(
        [np.concatenate(
            [env_visit_counts[j][tuple(zip(*state_inds[k]))].reshape(-1, 1, 1)
             for j in range(config.num_agents)], axis=1)
            for k in range(config.num_agents)], axis=2)

    # how novel each agent considers all agents observations at every step
    novelties = np.power(np.maximum(samp_visit_counts, 1), -config.decay)
    return torch.tensor(novelties, device=device, dtype=torch.float32)


def get_intrinsic_rewards(novelties, config, intr_rew_rms,
                          update_irrms=False, active_envs=None, device='cpu'):
    if update_irrms:
        assert active_envs is not None
    intr_rews = []

    for i, exp_type in enumerate(config.explr_types):
        if exp_type == 0:  # independent
            intr_rews.append([novelties[:, ai, ai] for ai in range(config.num_agents)])
        elif exp_type == 1:  # min
            intr_rews.append([novelties[:, :, ai].min(axis=1)[0] for ai in range(config.num_agents)])
        elif exp_type == 2:  # covering
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                rew[rew > 0.0] += novelties[rew > 0.0, :, ai].mean(axis=1)
                rew[rew < 0.0] = 0.0
                type_rews.append(rew)
            intr_rews.append(type_rews)
        elif exp_type == 3:  # burrowing
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                rew[rew > 0.0] = 0.0
                rew[rew < 0.0] += novelties[rew < 0.0, :, ai].mean(axis=1)
                type_rews.append(rew)
            intr_rews.append(type_rews)
        elif exp_type == 4:  # leader-follow
            type_rews = []
            for ai in range(config.num_agents):
                rew = novelties[:, ai, ai] - novelties[:, :, ai].mean(axis=1)
                if ai == 0:
                    rew[rew > 0.0] = 0.0
                    rew[rew < 0.0] += novelties[rew < 0.0, :, ai].mean(axis=1)
                else:
                    rew[rew > 0.0] += novelties[rew > 0.0, :, ai].mean(axis=1)
                    rew[rew < 0.0] = 0.0
                type_rews.append(rew)
            intr_rews.append(type_rews)

    for i in range(len(config.explr_types)):
        for j in range(config.num_agents):
            if update_irrms:
                intr_rew_rms[i][j].update(intr_rews[i][j].cpu().numpy(), active_envs=active_envs)
            intr_rews[i][j] = intr_rews[i][j].to(device)
            norm_fac = torch.tensor(np.sqrt(intr_rew_rms[i][j].var),
                                    device=device, dtype=torch.float32)
            intr_rews[i][j] /= norm_fac

    return intr_rews


def make_parallel_env(config, seed):
    lock = multiprocessing.Lock()

    def get_env_fn(rank):
        def init_env():
            if config.env_type == 'gridworld':
                env = VectObsEnv(GridWorld(config.map_ind,
                                           seed=(seed * 1000),
                                           task_config=config.task_config,
                                           num_agents=config.num_agents,
                                           need_get=False,
                                           stay_act=True), l=3)
            elif config.env_type == 'pedsmove':
                env = create_ped_env(map=config.map_ind,
                                     leader_num=config.num_agents,
                                     group_size=config.group_size,
                                     maxStep=config.max_episode_length,
                                     frame_skip=config.frame_skip,
                                     seed=(seed * 1000))
            else:  # vizdoom
                raise Exception("该修改代码不支持Vizdoom环境!")
            return env

        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in
                          range(config.n_rollout_threads)])


def run(config, load_file=None):
    # torch.set_num_threads(1)
    env_descr = '{}_{}agents_task{}'.format(config.map_ind, config.num_agents,
                                            config.task_config)
    model_dir = Path('./models') / config.env_type / env_descr / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    data_dir = run_dir / 'data'
    os.makedirs(log_dir)
    os.makedirs(data_dir)
    logger = SummaryWriter(str(log_dir))
    save_params(config, run_dir)
    data_dict = defaultdict(list)
    critic_policy_data_dict = defaultdict(list)
    head_data_dict = defaultdict(list)
    agent_pos_dict = defaultdict(lambda: defaultdict(list))

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config, run_num)
    if config.nonlinearity == 'relu':
        nonlin = torch.nn.functional.relu
    elif config.nonlinearity == 'leaky_relu':
        nonlin = torch.nn.functional.leaky_relu
    if config.intrinsic_reward == 0:
        n_intr_rew_types = 0
        sep_extr_head = True
    else:
        n_intr_rew_types = len(config.explr_types)
        sep_extr_head = False
    n_rew_heads = n_intr_rew_types + int(sep_extr_head)

    if load_file is None:
        model = SAC.init_from_env(env,
                                  nagents=config.num_agents,
                                  tau=config.tau,
                                  hard_update_interval=config.hard_update,
                                  pi_lr=config.pi_lr,
                                  q_lr=config.q_lr,
                                  phi_lr=config.phi_lr,
                                  adam_eps=config.adam_eps,
                                  q_decay=config.q_decay,
                                  phi_decay=config.phi_decay,
                                  gamma_e=config.gamma_e,
                                  gamma_i=config.gamma_i,
                                  pol_hidden_dim=config.pol_hidden_dim,
                                  critic_hidden_dim=config.critic_hidden_dim,
                                  nonlin=nonlin,
                                  reward_scale=config.reward_scale,
                                  head_reward_scale=config.head_reward_scale,
                                  beta=config.beta,
                                  n_intr_rew_types=n_intr_rew_types,
                                  sep_extr_head=sep_extr_head)
    else:
        logging.warning("加载之前保存的模型文件来训练...")
        model = SAC.init_from_save(load_file,
                                   load_critic=True,
                                   load_ir=True)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 env.state_space,
                                 env.observation_space,
                                 env.action_space)
    intr_rew_rms = [[RunningMeanStd()
                     for i in range(config.num_agents)]
                    for j in range(n_intr_rew_types)]
    eps_this_turn = 0  # episodes so far this turn
    active_envs = np.ones(config.n_rollout_threads)  # binary indicator of whether env is active
    env_times = np.zeros(config.n_rollout_threads, dtype=int)
    env_ep_extr_rews = np.zeros(config.n_rollout_threads)
    env_extr_rets = np.zeros(config.n_rollout_threads)
    env_ep_intr_rews = [[np.zeros(config.n_rollout_threads) for i in range(config.num_agents)]
                        for j in range(n_intr_rew_types)]
    recent_ep_extr_rews = deque(maxlen=100)
    recent_ep_intr_rews = [[deque(maxlen=100) for i in range(config.num_agents)]
                           for j in range(n_intr_rew_types)]
    recent_ep_lens = deque(maxlen=100)
    recent_found_treasures = [deque(maxlen=100) for i in range(config.num_agents)]
    recent_tiers_completed = deque(maxlen=100)
    meta_turn_rets = []
    extr_ret_rms = [RunningMeanStd() for i in range(n_rew_heads)]
    t = 0
    steps_since_update = 0

    state, obs = env.reset()

    pbar = tqdm.tqdm(total=config.train_time)

    while t < config.train_time:
        model.prep_rollouts(device='cuda' if config.gpu_rollout else 'cpu')
        # convert to torch tensor
        torch_obs = apply_to_all_elements(obs, lambda x: torch.tensor(x, dtype=torch.float32,
                                                                      device='cuda' if config.gpu_rollout else 'cpu'))
        # get actions as torch tensors
        torch_agent_actions = model.step(torch_obs, explore=True)
        # convert actions to numpy arrays
        agent_actions = apply_to_all_elements(torch_agent_actions, lambda x: x.cpu().data.numpy())
        # rearrange actions to be per environment
        actions = [[ac[i] for ac in agent_actions] for i in range(int(active_envs.sum()))]

        next_state, next_obs, rewards, dones, infos = env.step(actions, env_mask=active_envs)

        # try:
        #     with timeout(seconds=1):
        #         next_state, next_obs, rewards, dones, infos = env.step(actions, env_mask=active_envs)
        # except (TimeoutError):
        #     # 为了防止环境崩溃引起训练的无故终止!
        #     logging.warning("Environment are broken...")
        #     env = make_parallel_env(config, run_num)
        #     state, obs = env.reset()
        #     idx = active_envs.astype(bool)
        #     env_ep_extr_rews[idx] = 0.0
        #     env_extr_rets[idx] = 0.0
        #     for i in range(n_intr_rew_types):
        #         for j in range(config.num_agents):
        #             env_ep_intr_rews[i][j][idx] = 0.0
        #     env_times = np.zeros(config.n_rollout_threads, dtype=int)
        #     state = apply_to_all_elements(state, lambda x: x[idx])
        #     obs = apply_to_all_elements(obs, lambda x: x[idx])
        #     continue

        for env_idx in range(len(infos)):
            pos_list = infos[env_idx]['visit_count_lookup']
            for j in range(len(pos_list)):
                agent_pos_dict[j][env_idx].append(pos_list[j])

        steps_since_update += int(active_envs.sum())
        if config.intrinsic_reward == 1:
            # if using state-visit counts, store state indices
            # shape = (n_envs, n_agents, n_inds)
            state_inds = np.array([i['visit_count_lookup'] for i in infos],
                                  dtype=int)
            state_inds_t = state_inds.transpose(1, 0, 2)
            novelties = get_count_based_novelties(env, state_inds_t, device='cpu')
            intr_rews = get_intrinsic_rewards(novelties, config, intr_rew_rms,
                                              update_irrms=True, active_envs=active_envs,
                                              device='cpu')
            intr_rews = apply_to_all_elements(intr_rews, lambda x: x.numpy().flatten())
        else:
            intr_rews = None
            state_inds = None
            state_inds_t = None

        replay_buffer.push(state, obs, agent_actions, rewards, next_state, next_obs, dones,
                           state_inds=state_inds)
        env_ep_extr_rews[active_envs.astype(bool)] += np.array(rewards)
        env_extr_rets[active_envs.astype(bool)] += np.array(rewards) * config.gamma_e ** (
            env_times[active_envs.astype(bool)])
        env_times += active_envs.astype(int)
        if intr_rews is not None:
            for i in range(n_intr_rew_types):
                for j in range(config.num_agents):
                    env_ep_intr_rews[i][j][active_envs.astype(bool)] += intr_rews[i][j]
        over_time = env_times >= config.max_episode_length
        full_dones = np.zeros(config.n_rollout_threads)
        for i, env_i in enumerate(np.where(active_envs)[0]):
            full_dones[env_i] = dones[i]
        need_reset = np.logical_or(full_dones, over_time)
        # create masks ONLY for active envs
        active_over_time = env_times[active_envs.astype(bool)] >= config.max_episode_length
        active_need_reset = np.logical_or(dones, active_over_time)
        if any(need_reset):
            # reset any environments that are past the max number of time steps or done
            state, obs = env.reset(need_reset=need_reset)
        else:
            state, obs = next_state, next_obs
        for env_i in np.where(need_reset)[0]:
            recent_ep_extr_rews.append(env_ep_extr_rews[env_i])
            meta_turn_rets.append(env_extr_rets[env_i])
            if intr_rews is not None:
                for j in range(n_intr_rew_types):
                    for k in range(config.num_agents):
                        # record intrinsic rewards per step (so we don't confuse shorter episodes with less intrinsic rewards)
                        recent_ep_intr_rews[j][k].append(env_ep_intr_rews[j][k][env_i] / env_times[env_i])
                        env_ep_intr_rews[j][k][env_i] = 0

            recent_ep_lens.append(env_times[env_i])
            env_times[env_i] = 0
            env_ep_extr_rews[env_i] = 0
            env_extr_rets[env_i] = 0
            eps_this_turn += 1

            if eps_this_turn + active_envs.sum() - 1 >= config.metapol_episodes:
                active_envs[env_i] = 0

        for i in np.where(active_need_reset)[0]:
            for j in range(config.num_agents):
                # len(infos) = number of active envs
                recent_found_treasures[j].append(infos[i]['n_found_treasures'][j])
            if config.env_type == 'gridworld':
                recent_tiers_completed.append(infos[i]['tiers_completed'])

        if eps_this_turn >= config.metapol_episodes:
            if not config.uniform_heads and n_rew_heads > 1:
                meta_turn_rets = np.array(meta_turn_rets)
                if all(errms.count < 1 for errms in extr_ret_rms):
                    for errms in extr_ret_rms:
                        errms.mean = meta_turn_rets.mean()
                extr_ret_rms[model.curr_pol_heads[0]].update(meta_turn_rets)
                for i in range(config.metapol_updates):
                    model.update_heads_onpol(meta_turn_rets, extr_ret_rms, logger=logger, data_dict=head_data_dict)
            pol_heads = model.sample_pol_heads(uniform=config.uniform_heads)
            model.set_pol_heads(pol_heads)
            eps_this_turn = 0
            meta_turn_rets = []
            active_envs = np.ones(config.n_rollout_threads)

        if any(need_reset):  # reset returns state and obs for all envs, so make sure we're only looking at active
            state = apply_to_all_elements(state, lambda x: x[active_envs.astype(bool)])
            obs = apply_to_all_elements(obs, lambda x: x[active_envs.astype(bool)])

        if (len(replay_buffer) >= max(config.batch_size,
                                      config.steps_before_update) and
                (steps_since_update >= config.steps_per_update)):
            steps_since_update = 0
            # print('Updating at time step %i' % t)
            model.prep_training(device='cuda' if config.use_gpu else 'cpu')

            for u_i in range(config.num_updates):
                sample = replay_buffer.sample(config.batch_size,
                                              to_gpu=config.use_gpu,
                                              state_inds=(config.intrinsic_reward == 1))

                if config.intrinsic_reward == 0:  # no intrinsic reward
                    intr_rews = None
                    state_inds = None
                else:
                    sample, state_inds = sample
                    novelties = get_count_based_novelties(
                        env, state_inds,
                        device='cuda' if config.use_gpu else 'cpu')
                    intr_rews = get_intrinsic_rewards(novelties, config, intr_rew_rms,
                                                      update_irrms=False,
                                                      device='cuda' if config.use_gpu else 'cpu')

                model.update_critic(sample, logger=logger, intr_rews=intr_rews, data_dict=critic_policy_data_dict)
                model.update_policies(sample, logger=logger, data_dict=critic_policy_data_dict)
                model.update_all_targets()
            if len(recent_ep_extr_rews) > 10:
                logger.add_scalar('episode_rewards/extrinsic/mean',
                                  np.mean(recent_ep_extr_rews), t)
                logger.add_scalar('episode_lengths/mean',
                                  np.mean(recent_ep_lens), t)
                data_dict["timestep"].append(t)
                data_dict["episode_rewards/extrinsic/mean"].append(np.mean(recent_ep_extr_rews))
                data_dict["episode_lengths/mean"].append(np.mean(recent_ep_lens))
                if config.intrinsic_reward == 1:
                    for i in range(n_intr_rew_types):
                        for j in range(config.num_agents):
                            logger.add_scalar('episode_rewards/intrinsic%i_agent%i/mean' % (i, j),
                                              np.mean(recent_ep_intr_rews[i][j]), t)
                            data_dict['episode_rewards/intrinsic%i_agent%i/mean' % (i, j)].append(
                                np.mean(recent_ep_intr_rews[i][j]))
                for i in range(config.num_agents):
                    logger.add_scalar('agent%i/n_found_exit' % i, np.mean(recent_found_treasures[i]), t)
                    data_dict['agent%i/n_found_exit' % i].append(np.mean(recent_found_treasures[i]))

                logger.add_scalar('total_n_found_exit',
                                  sum(np.array(recent_found_treasures[i]) for i in range(config.num_agents)).mean(), t)
                data_dict['total_n_found_exit'].append(
                    sum(np.array(recent_found_treasures[i]) for i in range(config.num_agents)).mean())
                if config.env_type == 'gridworld':
                    logger.add_scalar('tiers_completed', np.mean(recent_tiers_completed), t)

        if t % config.save_interval == 0:
            model.prep_training(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_%isteps.pt' % (t + 1)))
            model.save(run_dir / 'model.pt')

            pd.DataFrame(data_dict).to_excel(data_dir / "main.xlsx", index=False)
            pd.DataFrame(critic_policy_data_dict).to_excel(data_dir / "cp_data.xlsx", index=False)
            pd.DataFrame(head_data_dict).to_excel(data_dir / "head.xlsx", index=False)

            with open(data_dir / "agent_pos.pkl", "wb") as f:
                dill.dump(agent_pos_dict, f)


        t += active_envs.sum()
        pbar.update(active_envs.sum())
    model.prep_training(device='cpu')
    model.save(run_dir / 'model.pt')
    logger.close()
    pd.DataFrame(data_dict).to_excel(data_dir / "main.xlsx", index=False)
    pd.DataFrame(critic_policy_data_dict).to_excel(data_dir / "cp_data.xlsx", index=False)
    pd.DataFrame(head_data_dict).to_excel(data_dir / "head.xlsx", index=False)

    with open(data_dir / "agent_pos.pkl", "wb") as f:
        dill.dump(agent_pos_dict, f)

    env.close(force=(config.env_type == 'vizdoom'))


from third_party.multi_explore.params.gridworld import params1 as p
from third_party.multi_explore.params.ped import params1 as ped_p

if __name__ == '__main__':
    config = ped_p.Params("map_09", 10, 4)
    # config.args.model_name = "one_icm_test"
    #config.args = ped_p.debug_mode(config.args)
    #config.args.train_time = 200
    #run(config.args)

    # for i in range(2):
    #     config = ped_p.icm_compare_test(params.args)
    #     run(config)


    config.args.model_name = strf_now_time() + "exp_test"
    for i in range(5):
        config.args = p.change_explore_type_exp(config.args)
        run(config.args)
