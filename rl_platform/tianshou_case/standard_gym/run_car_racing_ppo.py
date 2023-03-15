import datetime
import logging
import os
import pprint
from typing import Optional, Tuple, Union

import gym

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy, ICMPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic

import tianshou as ts

import sys

from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
from torch.optim import Adam, Optimizer

from rl_platform.tianshou_case.net.r_network import RNetwork
from rl_platform.tianshou_case.net.standard_net import CarRacingPolicyHead, CarRacingICMHead
from rl_platform.tianshou_case.standard_gym.wrapper import create_car_racing_env, CarRewardType
from rl_platform.tianshou_case.third_party import r_network_training
from rl_platform.tianshou_case.third_party.episodic_memory import EpisodicMemory

sys.path.append(r"D:\projects\python\PedestrainSimulationModule")

parallel_env_num = 8
test_env_num = 4
episode_per_test = 4
lr, gamma, n_steps = 2.5e-4, 0.99, 3
buffer_size = int(600000 / parallel_env_num)
batch_size = 128
eps_train, eps_test = 0.2, 0.05
max_epoch = 150
step_per_epoch = 10000
step_per_collect = 1000
repeat_per_collect = 4
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
update_per_step = 0.1
hidden_size = 100

actor_lr = lr
set_device = "cuda"
env_name = "normal"
# icm parameters
use_icm = True
icm_hidden_size = 256
icm_lr_scale = 1e-3
icm_reward_scale = 0.1
icm_forward_loss_weight = 0.2
# EC parameters
use_episodic_memory = False
exploration_reward = "episodic_curiosity"  # episodic_curiosity,oracle
scale_task_reward = 1.0
scale_surrogate_reward = 5.0  # 5.0 for vizdoom in ec,指的是EC奖励的放大倍数
bonus_reward_additive_term = 0
exploration_reward_min_step = 0  # 用于在线训练，在多少步时加入EC的相关奖励
similarity_threshold = 0.5
target_image_shape = [96, 96, 4]  # [96, 96, 4个连续灰度图像的堆叠]
r_network_checkpoint = r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\vizdoom\checkpoints\VizdoomMyWayHome-v0_PPO_2023_03_11_01_35_53\r_network_weight_500.pt"
# EC online train parameters
use_EC_online_train = False

# 文件配置相关
task = "CarRacing_{}".format(env_name)
file_name = task + "_PPO_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def get_policy(env, optim=None):
    # state_shape = env.observation_space.shape
    action_shape = env.action_space.shape or env.action_space.n

    h, w, c = target_image_shape
    # net = DQN(**cfg.policy.model)
    net = CarRacingPolicyHead(c, h, w, device=set_device)

    if set_device == "cuda":
        net.cuda()

    actor = Actor(net, action_shape, hidden_sizes=[256, hidden_size], device=set_device)
    critic = Critic(net, hidden_sizes=[256, hidden_size], device=set_device)
    optim = torch.optim.Adam(
        ActorCritic(actor, critic).parameters(), lr=lr, eps=eps_train
    )

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

    if use_icm:
        logging.warning(u"使用了ICM机制!")
        feature_net = CarRacingICMHead(c, h, w, device=set_device)
        if set_device == "cuda":
            feature_net.cuda()

        action_dim = np.prod(action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net,
            feature_dim,
            action_dim,
            hidden_sizes=[icm_hidden_size, 100],
            device=set_device,
        )
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=actor_lr)
        policy = ICMPolicy(
            policy, icm_net, icm_optim, icm_lr_scale, icm_reward_scale,
            icm_forward_loss_weight
        ).to(set_device)

    return policy, optim


def _get_agent(
        env,
        agent_learn: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        file_path=None
) -> tuple[Union[ICMPolicy, PPOPolicy, BasePolicy], Union[Adam, Optimizer, None], None]:
    if agent_learn is None:
        # model
        agent_learn, optim = get_policy(env, optim)
    if file_path is not None:
        state_dict = torch.load(file_path, map_location=set_device)
        if type(state_dict) is dict:
            agent_learn.load_state_dict(state_dict["agent"])
        else:
            agent_learn.load_state_dict(state_dict)

    return agent_learn, optim, None


env_test = False

if use_episodic_memory:
    net = RNetwork(target_image_shape, device=set_device)
    if set_device == 'cuda':
        net = net.cuda()
    if r_network_checkpoint is not None:
        net = torch.load(r_network_checkpoint, "cuda")
        logging.warning(u"加载完成RNetwork的相关参数!")
    else:
        raise RuntimeError(u"必须指定训练好的的R-Network!")
    net.eval()  # 此处是为了batchnorm而加
    memory = EpisodicMemory(observation_shape=[512],
                            observation_compare_fn=net.embedding_similarity)
    if use_EC_online_train:
        r_trainer = r_network_training.RNetworkTrainer(
            net,
            observation_history_size=10000,
            training_interval=500,
            num_train_epochs=1,
            checkpoint_dir=file_name,
            device=set_device)


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    global env_test, net, memory

    def wrapped_env():
        if not env_test:
            env = create_car_racing_env(zero_reward=CarRewardType.ZERO_REWARD)
        else:
            env = create_car_racing_env(zero_reward=CarRewardType.RAW_REWARD)
        if use_episodic_memory:
            logging.warning(u"使用了EC机制!")
            from rl_platform.tianshou_case.third_party.single_curiosity_env_wrapper import CuriosityEnvWrapper

            env = CuriosityEnvWrapper(
                env,
                memory,
                net.embed_observation,
                target_image_shape,
                # r_net_trainer=r_trainer,
                scale_task_reward=scale_task_reward,
                scale_surrogate_reward=scale_surrogate_reward,
                exploration_reward_min_step=exploration_reward_min_step,
                test_mode=env_test
            )
        return env

    return wrapped_env()


def train(load_check_point=None):
    global env_test, parallel_env_num, test_env_num, buffer_size, batch_size, debug, step_per_collect, episode_per_test
    if debug:
        parallel_env_num, test_env_num, buffer_size = 2, 1, 10000
        step_per_collect = 10
        episode_per_test = 0
        batch_size = 16
    if __name__ == "__main__":
        # ======== Step 1: Environment setup =========
        train_envs = SubprocVectorEnv([_get_env for _ in range(parallel_env_num)])
        env_test = True
        test_envs = DummyVectorEnv([_get_env for _ in range(test_env_num)])

        # ======== Step 2: Agent setup =========
        policy, optim, agents = _get_agent(_get_env())

        if load_check_point is not None:
            load_data = torch.load(load_check_point, map_location=set_device)
            policy.load_state_dict(load_data["agent"])

        # ======== Step 3: Collector setup =========
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(buffer_size, len(train_envs)),
            exploration_noise=True
        )
        test_collector = Collector(policy, test_envs, exploration_noise=False)

        # ======== Step 4: Callback functions setup =========
        writer = SummaryWriter('log/' + file_name)
        logger = ts.utils.TensorboardLogger(writer)  # TensorBoard is supported!

        def save_best_fn(policy):
            model_save_path = os.path.join('log/' + file_name, "policy.pth")
            os.makedirs(os.path.join('log/' + file_name), exist_ok=True)
            save_data = policy.state_dict()
            torch.save(save_data, model_save_path)

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join('log/' + file_name, "checkpoint_{}.pth".format(epoch))
            save_data = {"agent": policy.state_dict(), "optim": optim.state_dict()}
            torch.save(save_data, ckpt_path)
            return ckpt_path

        # def stop_fn(mean_rewards):
        #     return mean_rewards >= 2500

        def train_fn(epoch, env_step):
            policy.set_eps(eps_train)

        def test_fn(epoch, env_step):
            policy.set_eps(eps_test)

        def reward_metric(rews):
            return rews[:]

        # ======== Step 5: Run the trainer =========
        result = onpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            repeat_per_collect=repeat_per_collect,
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

        pprint.pprint(result)


def test():
    global env_test
    env_test = True
    policy_path = r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\vizdoom\log\Vizdoom_normal_PPO_2023_03_11_23_01_22\checkpoint_100.pth"
    test_envs = DummyVectorEnv([_get_env for _ in range(1)])
    env = _get_env()
    policy, optim, agents = _get_agent(env, None,
                                       file_path=policy_path)
    policy.eval()
    collector = Collector(policy, test_envs)
    res = collector.collect(n_episode=2, render=1 / 36)
    pprint.pprint(res)


def icm_one_experiment():
    global use_icm
    # use_icm
    train()
    # not use_icm
    use_icm = False
    train()


debug = False

if __name__ == "__main__":
    icm_one_experiment()

    # train_if = True
    #
    # if train_if:
    #     train()
    # else:
    #     test()
