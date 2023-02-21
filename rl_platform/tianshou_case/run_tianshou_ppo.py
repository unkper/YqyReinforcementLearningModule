import copy
import datetime
import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, PPOPolicy, ICMPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net, ActorCritic, MLP

import pettingzoo as pet
import tianshou as ts
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

import sys

from ped_env.mdp import PedsVisionRLHandler
from rl_platform.tianshou_case.net.network import MarioICMFeatureHead, PedICMFeatureHead, PedPolicyHead
from rl_platform.tianshou_case.utils.wrapper import FrameStackWrapper

sys.path.append(r"D:\projects\python\PedestrainSimulationModule")

from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import map_09, map_10, map_08
from rl_platform.tianshou_case.utils.common import _get_agents

parallel_env_num = 5
test_env_num = 1
lr, gamma, n_steps = 2.5e-4, 0.99, 3
buffer_size = 200000 // parallel_env_num
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


icm_lr_scale = 1e-3
icm_reward_scale = 0.1
icm_forward_loss_weight = 0.2
icm_hidden_size = 128

set_device = "cuda"


def get_policy(env, optim=None):

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    state_shape = (4, ) + state_shape

    from rl_platform.tianshou_case.net.network import MLPNetwork

    # net = MLPNetwork(
    #     state_dim=env.observation_space.shape[0],
    #     output_dim=128,
    #     hidden_dim=[128, 128, 128],
    #     device=set_device
    # )
    net = PedPolicyHead(
        *state_shape,
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

    if icm_lr_scale > 0:
        feature_net = PedICMFeatureHead(*state_shape,
                                        device=set_device)
        if set_device == "cuda":
            feature_net.cuda()

        action_dim = np.prod(action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net.net,
            feature_dim,
            action_dim,
            hidden_sizes=[icm_hidden_size],
            device=set_device,
        )
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=lr)
        policy = ICMPolicy(
            policy, icm_net, icm_optim, icm_lr_scale, icm_reward_scale,
            icm_forward_loss_weight
        ).to(set_device)

    return policy, optim


def _get_agents(
        agent_learn: Optional[BasePolicy] = None,
        agent_count: int = 1,
        optim: Optional[torch.optim.Optimizer] = None,
        file_path=None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    if agent_learn is None:
        # model
        agent_learn, optim = get_policy(env, optim)

    agents = [copy.deepcopy(agent_learn) for _ in range(agent_count)]
    if file_path is not None:
        state_dicts = torch.load(file_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        for ag, state_dict in zip(agents, state_dicts.values()):
            ag.load_state_dict(state_dict)
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


train_map = map_10
agent_num_map = 4

train_if = False  # 是否采用test模式，即在icm模式下采用奖励模型来评判


def _get_env():
    global train_if, max_step
    """This function is needed to provide callables for DummyVectorEnv."""
    env = FrameStackWrapper(PedsMoveEnv(train_map, person_num=agent_num_map, group_size=(1, 1), random_init_mode=True,
                                        maxStep=max_step, disable_reward=train_if, person_handler=PedsVisionRLHandler))
    env = pet.utils.parallel_to_aec(env)
    return PettingZooEnv(env)


def train(load_check_point=None, debug=False):
    global parallel_env_num, test_env_num, train_if, max_step, icm_lr_scale, buffer_size, batch_size
    if __name__ == "__main__":
        # ======== Step 1: Environment setup =========
        if debug:
            parallel_env_num, test_env_num = 3, 1
            max_step = 1000
            buffer_size = 200
            batch_size = 12
            #icm_lr_scale = 0
            train_envs = SubprocVectorEnv([_get_env for _ in range(parallel_env_num)])
            train_if = False
            test_envs = DummyVectorEnv([_get_env for _ in range(test_env_num)])
        else:
            train_envs = SubprocVectorEnv([_get_env for _ in range(parallel_env_num)])
            train_if = False
            test_envs = DummyVectorEnv([_get_env for _ in range(test_env_num)])

        # seed
        # seed = 25680
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # train_envs.seed(seed)
        # test_envs.seed(seed)

        # ======== Step 2: Agent setup =========
        policy, optim, agents = _get_agents(agent_count=agent_num_map)

        if load_check_point is not None:
            load_data = torch.load(load_check_point, map_location="cuda" if torch.cuda.is_available() else "cpu")
            for agent in agents:
                policy.policies[agent].load_state_dict(load_data[agent])

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
        task = "PedsMoveEnv_{}_{}".format(train_map.name, agent_num_map)
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

        # def reward_metric(rews):
        #     return rews[:, 1]

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
            #reward_metric=reward_metric,
            logger=logger
        )

        print(result)


def test():
    episode = 5
    # test_envs = DummyVectorEnv([_get_env for _ in range(1)])
    env = _get_env()
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
    train(debug=True)
    # test()

    # python run_tianshou_ppo.py --file=D:\projects\python\PedestrainSimulationModule\rl_platform\tianshou_case\log\PedsMoveEnv_map_10_40_PPO_2022_12_24_01_48_33\checkpoint_17.pth
