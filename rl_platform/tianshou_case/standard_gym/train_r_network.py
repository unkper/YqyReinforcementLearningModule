import os.path
from datetime import datetime

import gym
import torch
from tensorboardX import SummaryWriter
from vizdoom import gym_wrapper  # noqa
from tianshou.data.collector import Collector

from rl_platform.tianshou_case.net.r_network import RNetwork
from rl_platform.tianshou_case.third_party import r_network_training
from rl_platform.tianshou_case.third_party.single_curiosity_env_wrapper import resize_observation
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


def make_env():
    env = create_car_racing_env(zero_reward=CarRewardType.ZERO_REWARD)
    #env = create_walker_env()
    return env


def train(file = None):
    global set_device
    vec_env = make_env()
    writer = SummaryWriter(file_name)

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
        done = False
        obs = vec_env.reset()
        while not done:
            obs, rew, terminated, info = vec_env.step(vec_env.action_space.sample())
            #obs = resize_observation(obs, target_image_shape)
            r_trainer.on_new_observation(obs, rew, done, info)
            # pprint.pprint(rew)
            done = terminated
            if i >= total_feed_step:
                break
            pbar.update(1)
            i += 1


if __name__ == '__main__':
    #path = r"/rl_platform/tianshou_case/vizdoom/checkpoints/VizdoomMyWayHome-v0_PPO_2023_03_11_01_35_53\r_network_weight_500.pt"
    train()
