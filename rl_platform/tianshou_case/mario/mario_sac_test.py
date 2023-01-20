import argparse
import datetime
import os
import pprint
from typing import Optional, Tuple

import gym
import tianshou as ts

import gym_super_mario_bros
import numpy as np
import torch
from ding.envs import DingEnvWrapper, MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnEnv
from nes_py.wrappers import JoypadSpace
from tianshou.env import DummyVectorEnv
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DiscreteSACPolicy, ICMPolicy, BasePolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

from rl_platform.tianshou_case.mario.mario_model import DQN, TDQN

scale_obs = 0
buffer_size = 100000
actor_lr, critic_lr = 1e-5, 1e-5
gamma, tau, alpha = 0.99, 0.005, 0.05
n_step = 3
auto_alpha = False
alpha_lr = 3e-4
epoch = 100
step_per_epoch = 100000
step_per_collect = 10
epoch_per_test = 5
update_per_epoch = 0.1
batch_size, hidden_size = 64, 512
training_num, test_num = 10, 10
rew_norm = False
device = "cuda" if torch.cuda.is_available() else "cpu"
frames_stack = 4
icm_lr_scale = 0.
icm_reward_scale = 0.01
icm_forward_loss_weight = 0.2

env_name = "SuperMarioBros-1-1-v0"
action_type = [["right"], ["right", "A"]]

from .mario_dqn_config import mario_dqn_config as cfg

def get_policy(env, optim=None):
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net = DQN(**cfg.policy.model)
    if torch.cuda.is_available():
        net.cuda()

    actor = Actor(net, action_shape, device=device, softmax_output=False)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic1 = Critic(net, last_size=action_shape, device=device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net, last_size=action_shape, device=device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    # define policy
    if auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau,
        gamma,
        alpha,
        estimation_step=n_step,
        reward_normalization=rew_norm,
    ).to(device)
    if icm_lr_scale > 0:
        feature_net = TDQN(
            *state_shape, action_shape, device, features_only=True
        )
        action_dim = np.prod(action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net.net,
            feature_dim,
            action_dim,
            hidden_sizes=[hidden_size],
            device=device,
        )
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=actor_lr)
        policy = ICMPolicy(
            policy, icm_net, icm_optim, icm_lr_scale, icm_reward_scale,
            icm_forward_loss_weight
        ).to(device)

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
        agent_learn.load_state_dict(state_dict)
    return agent_learn, optim, None


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    def wrapped_mario_env():
        return DingEnvWrapper(
                JoypadSpace(gym_super_mario_bros.make(env_name), action_type),
                cfg={
                    'env_wrapper': [
                        lambda env: MaxAndSkipWrapper(env, skip=4),
                        lambda env: WarpFrameWrapper(env, size=84),
                        lambda env: ScaledFloatFrameWrapper(env),
                        lambda env: FrameStackWrapper(env, n_frames=4),
                        lambda env: EvalEpisodeReturnEnv(env),
                    ]
                }
            )
    return wrapped_mario_env()

def train(load_check_point=None):
    if __name__ == "__main__":
        # ======== Step 1: Environment setup =========
        train_envs = DummyVectorEnv([_get_env for _ in range(training_num)])
        test_envs = DummyVectorEnv([_get_env for _ in range(test_num)])

        # seed
        seed = 21343
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)

        # ======== Step 2: Agent setup =========
        policy, optim, agents = _get_agent(agent_count=1)

        if load_check_point is not None:
            load_data = torch.load(load_check_point, map_location="cuda" if torch.cuda.is_available() else "cpu")
            policy.load_state_dict(load_data["agent"])
            optim.load_state_dict(load_data["optim"])

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
        file_name = task + "_SAC_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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
            return mean_rewards >= 3000

        # def train_fn(epoch, env_step):
        #     policy.set_eps(eps_train)
        #
        # def test_fn(epoch, env_step):
        #     policy.set_eps(eps_test)

        def reward_metric(rews):
            return rews[:]

        # ======== Step 5: Run the trainer =========
        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            episode_per_test=epoch_per_test,
            batch_size=batch_size,
            # train_fn=train_fn,
            # test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            update_per_step=update_per_epoch,
            test_in_train=False,
            reward_metric=reward_metric,
            logger=logger
        )

        pprint.pprint(result)

def test():
    policy_path = r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\mario\log\Mario_SuperMarioBros-1-1-v0_DQN_2023_01_18_12_00_34\policy.pth"
    test_envs = DummyVectorEnv([_get_env for _ in range(1)])
    env = _get_env()
    policy, optim, agents = _get_agent(None, 8,
                                       file_path=policy_path)
    policy.eval()
    collector = Collector(policy, test_envs)
    collector.collect(n_episode=5, render=1 / 36)


if __name__ == "__main__":
    train()
