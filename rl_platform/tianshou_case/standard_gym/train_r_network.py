import os.path
import pprint
from datetime import datetime

import gym
import torch
from tensorboardX import SummaryWriter
from tianshou.data import ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import RandomPolicy
from vizdoom import gym_wrapper  # noqa
from tianshou.data.collector import Collector

from rl_platform.tianshou_case.third_party.r_network_training import train_r_network_with_collector
from rl_platform.tianshou_case.utils.dummy_policy import DummyPolicy
from wrapper import create_walker_env, create_car_racing_env, RewardType

env_name = "CarRacing_v3"
set_device = "cuda"
task = "{}".format(env_name)
file_name = os.path.abspath(os.path.join("r_network", task + "_PPO_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
total_feed_step = 200000
observation_history_size = 20000
training_interval = 2000
num_train_epochs = 30
batch_size = 128
# target_image_shape = [120, 160, 3]
target_image_shape = [96, 96, 4]
step_interval = 500
train_env_num = 10


def make_env():
    env = create_car_racing_env(zero_reward=RewardType.RAW_REWARD)
    # env = create_walker_env()
    return env


debug = True


def test_collector():
    policy = DummyPolicy(make_env().action_space)
    train_envs = SubprocVectorEnv([make_env for _ in range(5)])
    buffer = VectorReplayBuffer(100, len(train_envs))
    collector = Collector(policy, train_envs, buffer=buffer)
    collector.collect(n_step=2)
    pprint.pprint(buffer.sample(2))

from easydict import EasyDict

if __name__ == '__main__':
    dic = EasyDict(globals())
    train_r_network_with_collector(make_env, file_name, dic)
    # path = r"/rl_platform/tianshou_case/vizdoom/checkpoints/VizdoomMyWayHome-v0_PPO_2023_03_11_01_35_53\r_network_weight_500.pt"
    # train()
    #train(r"D:\Projects\python\PedestrainSimlationModule\rl_platform\tianshou_case\standard_gym\r_network\CarRacing_v3_PPO_2023_03_16_00_07_57\r_network_weight_150.pt")
