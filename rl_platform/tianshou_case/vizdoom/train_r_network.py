import os.path
from datetime import datetime

import gym
import vizdoom
from vizdoom import gym_wrapper
import tqdm
from tensorboardX import SummaryWriter

from rl_platform.tianshou_case.net.r_network import RNetwork
from rl_platform.tianshou_case.third_party import episodic_memory, r_network_training
from rl_platform.tianshou_case.utils.single_curiosity_env_wrapper import CuriosityEnvWrapper
from rl_platform.tianshou_case.vizdoom.vizdoom_env_wrapper import VizdoomEnvWrapper

env_name = "VizdoomMyWayHome-v0"
set_device = "cuda"
task = "{}".format(env_name)
file_name = os.path.join("r_network", task + "_PPO_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
total_feed_step = 100000
observation_history_size = 10000
training_interval = 500
num_train_epochs = 1
embedding_size = 16
memory_capacity = 1000
target_image_shape = [14, 14, 1]


def train():
    global set_device
    vec_env = VizdoomEnvWrapper(gym.make(env_name))
    writer = SummaryWriter(file_name)

    net = RNetwork(vec_env.observation_space, device=set_device)
    if set_device == 'cuda':
        net = net.cuda()
    r_trainer = r_network_training.RNetworkTrainer(
        net,
        observation_history_size=observation_history_size,
        training_interval=training_interval,
        num_train_epochs=num_train_epochs,
        checkpoint_dir=file_name,
        device=set_device,
        writer=writer)

    for i in tqdm.tqdm(range(total_feed_step), desc="r-network training:"):
        done = False
        obs = vec_env.reset()
        while not done:
            obs, rew, terminated, info = vec_env.step(vec_env.action_space.sample())
            r_trainer.on_new_observation(obs, rew, done, info)
            # pprint.pprint(rew)
            done = terminated


if __name__ == '__main__':
    train()
