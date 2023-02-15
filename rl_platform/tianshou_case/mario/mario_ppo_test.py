import copy
import datetime
import os
import pprint
from typing import Optional, Tuple

import gym
import gym_super_mario_bros
import numpy as np
import torch
from ding.envs import DingEnvWrapper, MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnEnv
from nes_py.wrappers import JoypadSpace
from tensorboardX import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, PPOPolicy, ICMPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils.net.common import Net, ActorCritic

import pettingzoo as pet
import tianshou as ts

import sys

from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

from rl_platform.tianshou_case.mario.mario_dqn_config import mario_dqn_config, SIMPLE_MOVEMENT
from rl_platform.tianshou_case.mario.mario_model import DQN
from rl_platform.tianshou_case.net.network import MarioICMFeatureHead, MarioPolicyHead
from rl_platform.tianshou_case.utils.wrapper import MarioRewardWrapper

sys.path.append(r"D:\projects\python\PedestrainSimulationModule")


parallel_env_num = 10
lr, gamma, n_steps = 2.5e-4, 0.99, 3
buffer_size = 200000 / parallel_env_num
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

actor_lr = lr
set_device = "cuda"

cfg = mario_dqn_config
env_name = "SuperMarioBros-1-1-v0"
action_type = [["right"], ["right", "A"], ["right", "B"]]

icm_hidden_size = 256
icm_lr_scale = 1e-3
icm_reward_scale = 0.1
icm_forward_loss_weight = 0.2

def get_policy(env, optim=None):
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape or env.action_space.n

    #net = DQN(**cfg.policy.model)
    net = MarioPolicyHead(*state_shape, device=set_device)

    if set_device == "cuda":
        net.cuda()

    actor = Actor(net, env.action_space.n, device=set_device, softmax_output=False)
    critic = Critic(net, device=set_device)
    optim = torch.optim.Adam(
        ActorCritic(actor, critic).parameters(), lr=lr, eps = eps_train
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

    if icm_lr_scale > 0:
        feature_net = MarioICMFeatureHead(*state_shape, device=set_device)
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
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=actor_lr)
        policy = ICMPolicy(
            policy, icm_net, icm_optim, icm_lr_scale, icm_reward_scale,
            icm_forward_loss_weight
        ).to(set_device)

    return policy, optim


def _get_agent(
        agent_learn: Optional[BasePolicy] = None,
        agent_count: int = 1,
        optim: Optional[torch.optim.Optimizer] = None,
        file_path=None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    if agent_learn is None:
        # model
        agent_learn, optim = get_policy(env, optim)
    if file_path is not None:
        state_dict = torch.load(file_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if type(state_dict) is dict:
            agent_learn.load_state_dict(state_dict["agent"])
        else:
            agent_learn.load_state_dict(state_dict)

    return agent_learn, optim, None

env_test = False

def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    global env_test
    def wrapped_mario_env():
        wrappers = [
            lambda env: MaxAndSkipWrapper(env, skip=4),
            lambda env: WarpFrameWrapper(env, size=84),
            lambda env: ScaledFloatFrameWrapper(env),
            lambda env: FrameStackWrapper(env, n_frames=4),
            lambda env: EvalEpisodeReturnEnv(env),
        ]
        if not env_test:
            wrappers.append(lambda env: MarioRewardWrapper(env))  # 为了验证ICM机制的有效性而加

        return DingEnvWrapper(
            JoypadSpace(gym_super_mario_bros.make(env_name), action_type),
            cfg={
                'env_wrapper': wrappers
            }
        )
    return wrapped_mario_env()

def _get_test_env():
    return gym_super_mario_bros.make(env_name)


def train(load_check_point=None):
    global env_test
    if __name__ == "__main__":
        # ======== Step 1: Environment setup =========
        train_envs = ShmemVectorEnv([_get_env for _ in range(parallel_env_num)])
        env_test = True
        test_envs = ShmemVectorEnv([_get_env for _ in range(test_env_num)])

        # seed
        # seed = 21343
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # train_envs.seed(seed)
        # test_envs.seed(seed)

        # ======== Step 2: Agent setup =========
        policy, optim, agents = _get_agent(agent_count=1)

        if load_check_point is not None:
            load_data = torch.load(load_check_point, map_location="cuda" if torch.cuda.is_available() else "cpu")
            policy.load_state_dict(load_data["agent"])
            # for i in range(len(optim)):
            #     optim[i].load_state_dict(load_data["optim"][i])

        # ======== Step 3: Collector setup =========
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(buffer_size, len(train_envs)),
            exploration_noise=True
        )
        test_collector = Collector(policy, test_envs, exploration_noise=True)

        # train_collector.collect(n_step=batch_size * 10)  # batch size * training_num

        # ======== Step 4: Callback functions setup =========
        task = "Mario_{}".format(env_name)
        file_name = task + "_PPO_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logger = ts.utils.TensorboardLogger(SummaryWriter('log/' + file_name))  # TensorBoard is supported!

        def save_best_fn(policy):
            model_save_path = os.path.join('log/' + file_name, "policy.pth")
            os.makedirs(os.path.join('log/' + file_name), exist_ok=True)
            save_data = policy.state_dict()
            torch.save(save_data, model_save_path)

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join('log/' + file_name, "checkpoint_{}.pth".format(epoch))
            save_data = {}
            save_data["agent"] = policy.state_dict()
            save_data["optim"] = optim.state_dict()
            torch.save(save_data, ckpt_path)
            return ckpt_path

        def stop_fn(mean_rewards):
            return mean_rewards >= 2500

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
            repeat_per_collect = 4,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            # train_fn=train_fn,
            # test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            update_per_step=update_per_step,
            test_in_train=False,
            reward_metric=reward_metric,
            logger=logger
        )

        pprint.pprint(result)


def test():
    policy_path = r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\mario\log\Mario_SuperMarioBros-1-1-v0_PPO_2023_02_09_14_17_48\checkpoint_380.pth"
    test_envs = DummyVectorEnv([_get_env for _ in range(1)])
    env = _get_env()
    policy, optim, agents = _get_agent(None, 8,
                                       file_path=policy_path)
    policy.eval()
    collector = Collector(policy, test_envs)
    collector.collect(n_episode=5, render=1 / 36)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument("--test", type=bool, action='store_true', default=False)

    parmas = parser.parse_args()

    train_if = True

    if not train_if:
        train()
    else:
        test()
    # train(r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\mario\log\Mario_SuperMarioBros-1-2-v0_DQN_2023_01_17_19_22_53\checkpoint_23.pth")
    # test()

    # python run_tianshou.py --load_file=D:\projects\python\PedestrainSimulationModule\rl_platform\tianshou_case\log\PedsMoveEnv_map_10_40_DQN_2022_12_21_08_13_37\checkpoint_35.pth
