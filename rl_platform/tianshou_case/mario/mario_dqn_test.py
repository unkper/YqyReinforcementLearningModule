import copy
import datetime
import os
from typing import Optional, Tuple

import gym
import gym_super_mario_bros
import numpy as np
import torch
from ding.config import compile_config
from ding.envs import DingEnvWrapper, MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnEnv, SyncSubprocessEnvManager
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from nes_py.wrappers import JoypadSpace
from tensorboardX import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

import pettingzoo as pet
import tianshou as ts

import sys

from rl_platform.tianshou_case.mario.mario_dqn_config import mario_dqn_config
from rl_platform.tianshou_case.mario.mario_model import DQN

sys.path.append(r"D:\projects\python\PedestrainSimulationModule")


lr, gamma, n_steps = 1e-4, 0.99, 3
buffer_size = 200000
batch_size = 256
eps_train, eps_test = 0.2, 0.05
max_epoch = 100
max_step = 10000  # 环境的最大步数
step_per_epoch = max_step
step_per_collect = 50
episode_per_test = 5
update_per_step = 0.1
seed = 1
train_env_num, test_env_num = 10, 10

cfg = mario_dqn_config


def get_policy(env, optim=None):
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )

    # net = Net(
    #     state_shape=observation_space.shape
    #                 or observation_space.n,
    #     action_shape=env.action_space.shape or env.action_space.n,
    #     hidden_sizes=[128, 128, 128],
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # ).to("cuda" if torch.cuda.is_available() else "cpu")
    net = DQN(**cfg.policy.model)

    from rl_platform.tianshou_case.net.utils import layer_init

    #layer_init(net)

    if optim is None:
        optim = torch.optim.Adam(net.parameters(), lr=lr)
    agent_learn = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=gamma,
        estimation_step=n_steps,
        target_update_freq=320,
    )
    return agent_learn, optim


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

    agents = [agent_learn for _ in range(agent_count)]
    if file_path is not None:
        state_dicts = torch.load(file_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        for ag, state_dict in zip(agents, state_dicts.values()):
            ag.load_state_dict(state_dict)
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    def wrapped_mario_env():
        return DingEnvWrapper(
            JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-4-v0"), [["right"], ["right", "A"]]),
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
        train_envs = DummyVectorEnv([_get_env for _ in range(train_env_num)])
        test_envs = DummyVectorEnv([_get_env for _ in range(test_env_num)])

        # seed
        seed = 21343
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)

        # ======== Step 2: Agent setup =========
        policy, optim, agents = _get_agents(agent_count=1)

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
            exploration_noise=True,
            reset_when_all_done=True
        )
        test_collector = Collector(policy, test_envs, exploration_noise=True, reset_when_all_done=True)

        train_collector.collect(n_step=batch_size * 10)  # batch size * training_num

        # ======== Step 4: Callback functions setup =========
        task = "Mario_{}".format("1-4")
        file_name = task + "_DQN_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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

        def stop_fn(mean_rewards):
            return mean_rewards >= 0.6

        def train_fn(epoch, env_step):
            for agent in agents:
                policy.policies[agent].set_eps(eps_train)

        def test_fn(epoch, env_step):
            for agent in agents:
                policy.policies[agent].set_eps(eps_test)

        def reward_metric(rews):
            return rews[:, 1]

        # ======== Step 5: Run the trainer =========
        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            update_per_step=update_per_step,
            test_in_train=False,
            reward_metric=reward_metric,
            logger=logger
        )

        print(result)


def test():
    test_envs = DummyVectorEnv([_get_env for _ in range(1)])
    env = _get_env()
    policy = get_policy(env)
    policy, optim, agents = _get_agents(policy, 8,
                                        file_path=r"/rl_platform/log/PedsMoveEnv_DQN_2022_12_17_14_54_15/policy.pth")
    collector = Collector(policy, test_envs)
    collector.collect(n_episode=5, render=1 / 36)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_file", type=str, default=None)

    parmas = parser.parse_args()

    train(parmas.load_file)
    # test()

    # python run_tianshou.py --load_file=D:\projects\python\PedestrainSimulationModule\rl_platform\tianshou_case\log\PedsMoveEnv_map_10_40_DQN_2022_12_21_08_13_37\checkpoint_35.pth
