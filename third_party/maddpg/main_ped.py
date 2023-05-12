import os
import random

import tqdm

from ped_env.envs import PedsMoveEnv
from ped_env.interfaces.maddpg_interface import MADDPG_Wrapper

from MADDPG import MADDPG
import numpy as np
import torch as th
from tensorboardX import SummaryWriter
from params import scale_reward
import pandas as pd
import datetime

# do not render the scene
e_render = False
n_agents = 4

world = MADDPG_Wrapper(PedsMoveEnv("map_07", n_agents, discrete=False))


reward_record = []

sed = random.randint(0, 10000)
np.random.seed(sed)
th.manual_seed(sed)
world.seed(sed)
n_agents = world.env.num_agents
n_states = 16  # 状态空间的大小
n_actions = 2  # 动作空间的大小
capacity = 5000000 # Replay Buffer的大小
batch_size = 1024

n_episode = 500  # 总共运行多少个episode
max_steps = 10000
episodes_before_train = 20  #运行多少个episode后开始训练

win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

save_path = "./save/{}_peds_move".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
os.makedirs(save_path)
tf = SummaryWriter(logdir=save_path)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in tqdm.tqdm(range(n_episode)):
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        if i_episode % 100 == 0 and e_render:
            world.render()
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward, done, _ = world.step(action.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs
        if t % 100 == 0:
            c_loss, a_loss = maddpg.update_policy()
        if done:
            break
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward / n_agents)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')


    tf.add_scalar('Total reward', total_reward, maddpg.episode_done)


world.close()
df = pd.DataFrame({
    "mean_reward": reward_record,
})
df.to_excel(os.path.join(save_path, "data.xlsx"), index=False)
tf.close()