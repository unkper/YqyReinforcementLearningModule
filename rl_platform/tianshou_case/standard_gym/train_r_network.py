import os.path
import pprint
from datetime import datetime

import gym
import torch
from tensorboardX import SummaryWriter
from tianshou.data import ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy
from vizdoom import gym_wrapper  # noqa
from tianshou.data.collector import Collector

from rl_platform.tianshou_case.net.r_network import RNetwork
from rl_platform.tianshou_case.third_party import r_network_training
from rl_platform.tianshou_case.third_party.single_curiosity_env_wrapper import resize_observation
from rl_platform.tianshou_case.utils.dummy_policy import DummyPolicy
from wrapper import create_walker_env, create_car_racing_env, CarRewardType

env_name = "CarRacing_v3"
set_device = "cuda"
task = "{}".format(env_name)
file_name = os.path.join("r_network", task + "_PPO_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
total_feed_step = 200000
observation_history_size = 20000
training_interval = 20000
num_train_epochs = 50
batch_size = 128
#target_image_shape = [120, 160, 3]
target_image_shape = [96, 96, 4]
step_interval = 500
train_env_num = 5

def make_env():
    env = create_car_racing_env(zero_reward=CarRewardType.RAW_REWARD)
    #env = create_walker_env()
    return env


def train(file = None):
    global set_device
    writer = SummaryWriter(file_name)
    vec_env = make_env()

    policy = DummyPolicy(make_env().action_space)
    train_envs = DummyVectorEnv([make_env for _ in range(train_env_num)])
    buffer = VectorReplayBuffer(1000, len(train_envs))
    collector = Collector(policy, train_envs, buffer=buffer)

    net = RNetwork(vec_env.observation_space, device=set_device)
    if file is not None:
        net = torch.load(file, map_location="cuda")
    if set_device == 'cuda':
        net = net.cuda()
    r_trainer = r_network_training.RNetworkTrainer(
        net,
        observation_history_size=observation_history_size,
        training_interval=training_interval,
        num_train_epochs=num_train_epochs,
        checkpoint_dir=file_name,
        device=set_device,
        batch_size=batch_size,
        writer=writer)
    from tqdm import tqdm

    pbar = tqdm(total=total_feed_step, desc="r-network training:")
    i = 0

    while i < total_feed_step:
        collector.collect(n_step=step_interval)
        batch, _ = buffer.sample(step_interval)
        for j in range(step_interval):
            r_trainer.on_new_observation(batch.obs[j], batch.rew[j], batch.done[j], batch.info[j])
        pbar.update(step_interval)
        i += step_interval

def test_collector():
    policy = DummyPolicy(make_env().action_space)
    train_envs = DummyVectorEnv([make_env for _ in range(5)])
    buffer = VectorReplayBuffer(100, len(train_envs))
    collector = Collector(policy, train_envs, buffer=buffer)
    collector.collect(n_step=2)
    pprint.pprint(buffer.sample(2))


if __name__ == '__main__':
    #path = r"/rl_platform/tianshou_case/vizdoom/checkpoints/VizdoomMyWayHome-v0_PPO_2023_03_11_01_35_53\r_network_weight_500.pt"
    #train()
    train()
