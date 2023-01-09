import copy
import datetime
import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net, ActorCritic, MLP

import pettingzoo as pet
import tianshou as ts
from tianshou.utils.net.discrete import Actor, Critic

import sys
sys.path.append(r"D:\projects\python\PedestrainSimulationModule")

from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import map_08, map_10
from rl_platform.tianshou_case.utils.common import _get_agents

lr, gamma, n_steps = 2.5e-4, 0.99, 3
buffer_size = 200000
batch_size = 256
eps_train, eps_test = 0.2, 0.05
max_epoch = 500
max_step = 10000  # 环境的最大步数
step_per_epoch = max_step
step_per_collect = 1000
rew_norm = True
vf_coef = 0.25
ent_coef = 0.01
gae_lambda = 0.95
lr_decay = True
lr_scheduler = None
max_grad_norm = 0.5
eps_clip = 0.1
dual_clip = None
value_clip = 1
norm_adv = 1
recompute_adv = 0
episode_per_test = 5
update_per_step = 0.1
seed = 1
train_env_num, test_env_num = 10, 10

set_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_policy(env, optim=None):
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )

    from rl_platform.tianshou_case.net.network import MLPNetwork

    net = MLPNetwork(
        state_dim=env.observation_space.shape[0],
        output_dim=128,
        hidden_dim=[128, 128, 128],
        device=set_device
    )

    actor = Actor(net, env.action_space.n, device=set_device, softmax_output=False)
    critic = Critic(net, device=set_device)
    optim = torch.optim.Adam(
        ActorCritic(actor, critic).parameters(), lr=lr, eps=eps_train
    )

    # define policy
    def dist(p):
        return torch.distributions.Categorical(logits=p)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        reward_normalization=rew_norm,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=eps_clip,
        value_clip=value_clip,
        dual_clip=dual_clip,
        advantage_normalization=norm_adv,
        recompute_advantage=recompute_adv,
    ).to(set_device)

    return policy, optim


train_map = map_10
agent_num_map8 = 40


def _get_env(test=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = PedsMoveEnv(train_map, person_num=agent_num_map8, group_size=(1, 1), random_init_mode=True,
                      maxStep=max_step)
    if test:
        return env
    else:
        temp = pet.utils.parallel_to_aec(env)
    return PettingZooEnv(temp)


def train(load_check_point=None):
    if __name__ == "__main__":
        # ======== Step 1: Environment setup =========
        train_envs = DummyVectorEnv([_get_env for _ in range(train_env_num)])
        test_envs = DummyVectorEnv([_get_env for _ in range(test_env_num)])

        # seed
        seed = 25680
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)

        # ======== Step 2: Agent setup =========
        policy, optim, agents = _get_agents(_get_env(), agent_count=agent_num_map8, get_policy=get_policy)

        if load_check_point is not None:
            load_data = torch.load(load_check_point, map_location="cuda" if torch.cuda.is_available() else "cpu")
            for agent in agents:
                policy.policies[agent].load_state_dict(load_data[agent])
            optim.load_state_dict(load_data["optim"])

        # ======== Step 3: Collector setup =========
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(buffer_size, len(train_envs)),
            exploration_noise=True
        )
        test_collector = Collector(policy, test_envs, exploration_noise=True)

        train_collector.collect(n_step=batch_size * 10)  # batch size * training_num # use random policy

        # ======== Step 4: Callback functions setup =========
        task = "PedsMoveEnv_{}_{}".format(train_map.name, agent_num_map8)
        file_name = task + "_PPO_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logger = ts.utils.TensorboardLogger(SummaryWriter('log/' + file_name))  # TensorBoard is supported!

        def save_best_fn(policy):
            model_save_path = os.path.join('log/' + file_name, "policy.pth")
            os.makedirs(os.path.join('log/' + file_name), exist_ok=True)
            save_data = {}
            for agent in agents:
                save_data[agent] = policy.policies[agent].state_dict()
            torch.save(save_data, model_save_path)

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join('log/' + file_name, "checkpoint_{}.pth".format(epoch))
            save_data = {}
            for agent in agents:
                save_data[agent] = policy.policies[agent].state_dict()
            save_data["optim"] = optim.state_dict()
            torch.save(save_data, ckpt_path)
            return ckpt_path

        # def stop_fn(mean_rewards):
        #     return mean_rewards >= 0.6

        # def train_fn(epoch, env_step):
        #     for agent in agents:
        #         policy.policies[agent].set_eps(eps_train)
        #
        # def test_fn(epoch, env_step):
        #     for agent in agents:
        #         policy.policies[agent].set_eps(eps_test)

        def reward_metric(rews):
            return rews[:, 1]

        # ======== Step 5: Run the trainer =========
        result = onpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            repeat_per_collect=4,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            # train_fn=train_fn,
            # test_fn=test_fn,
            # stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            update_per_step=update_per_step,
            test_in_train=False,
            reward_metric=reward_metric,
            logger=logger
        )

        print(result)


def test():
    episode = 5
    # test_envs = DummyVectorEnv([_get_env for _ in range(1)])
    env = _get_env(True)
    policy, optim = get_policy(_get_env())

    file_path = r"D:\projects\python\PedestrainSimulationModule\rl_platform\tianshou_case\log" \
                r"\PedsMoveEnv_map_10_40_PPO_2022_12_22_16_38_53\policy.pth "
    if file_path is not None:
        state_dicts = torch.load(file_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        policy.load_state_dict(state_dicts[env.agents[0]])

    # collector = Collector(policy, test_envs)
    # collector.collect(n_episode=5, render=1 / 100)
    policy.eval()
    for i in range(episode):
        obs = env.reset()
        is_done = {"agent_1": False}
        while not all(is_done.values()):
            action = {}
            for agent in env.agents:
                batch = Batch(obs=[obs[agent]])
                act = policy(batch).act[0]
                action[agent] = act
            obs, reward, is_done, truncated, info = env.step(action)
            # env.render()

import argparse

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("file", type=str, default=None)
    # parser.add_argument("max_step", type=int, default=5000)
    # args = parser.parse_args()
    # train()
    train(r"D:\projects\python\PedestrainSimulationModule\rl_platform\tianshou_case\log\PedsMoveEnv_map_10_40_PPO_2023_01_01_21_24_54\checkpoint_14.pth")
    # test()

    #python run_tianshou_ppo.py --file=D:\projects\python\PedestrainSimulationModule\rl_platform\tianshou_case\log\PedsMoveEnv_map_10_40_PPO_2022_12_24_01_48_33\checkpoint_17.pth
