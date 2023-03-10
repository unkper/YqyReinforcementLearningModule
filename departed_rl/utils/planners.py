import shutil
import datetime
import os
import pickle
import time

import numpy as np
from tqdm import tqdm

from ped_env.pathfinder import AStarController
from departed_rl.agents.Agent import Agent
from departed_rl.utils.classes import Experience, Transition, make_parallel_env
from departed_rl.utils.functions import load_experience


def load_offline_train(env, data, lock=None):
    print("开始进行离线BC训练...")
    planner = AStarPlanner(env, load_mode=True)
    planner.load_experience(data, lock)
    return planner.experience

class AStarPlanner:
    def __init__(self, env, capacity=1e6, n_rol_threads=1, load_mode=False, use_random_policy=False, discrete=False):
        '''
        该类利用env进行仿真模拟，然后将收集到的trans压入experience中
        :param env:
        :param experience:
        '''
        self.env = env
        self.experience = Experience(capacity)
        self.init_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
        self.n_rol_threads = n_rol_threads
        self.random_policy = use_random_policy

        if not load_mode:
            self.controler = AStarController(env, use_random_policy, discrete=discrete) \
                if n_rol_threads == 1 else make_parallel_env(AStarController(env, use_random_policy, discrete=discrete), n_rol_threads)

    def planning(self, episodes=1):
        for epoch in tqdm(range(0, episodes, self.n_rol_threads)):
            step, starttime = 0, time.time()
            total_reward = 0.0
            obs = self.controler.reset()
            is_done = np.array([False])
            while not is_done.any():
                next_obs, reward, is_done, action = self.controler.step(obs)
                total_reward += np.mean(reward)
                if self.n_rol_threads == 1:
                    is_done = np.array(is_done)
                    trans = Transition(obs, action, reward, is_done, next_obs)
                    self.experience.push(trans)
                else:
                    for i in range(self.n_rol_threads):
                        trans = Transition(obs[i], action[i], reward[i], is_done[i], next_obs[i])
                        self.experience.push(trans)
                obs = next_obs
                step += 1#self.controler.frame_skipping
            endtime = time.time()
            # print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
            #      .format(self.env.listener.col_with_agent, self.env.listener.col_with_wall))
            print("奖励为{}!".format(total_reward))
            #print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime, step / (endtime - starttime)))

    def clear_experience(self):
        self.experience.clear()

    def save_experience(self):
        dir_name = "{}_exp".format(self.env.terrain.name)
        if self.random_policy:
            dir_name += "_random"
        pa = "./data/exp/" + dir_name
        if os.path.exists(pa):
            shutil.rmtree(pa)
        os.mkdir(pa)
        file_exp = open(os.path.join(pa, "experience.pkl"),"wb")
        pickle.dump(self.experience,file_exp,0)
        file_desc = open(os.path.join(pa, "desc.txt"), "w+")
        file_desc.write("person_num:{}\n".format(self.env.person_num))
        file_desc.write("group_size:{}\n".format(self.env.group_size))
        file_desc.write("random_init:{}\n".format(self.env.random_init_mode))
        file_desc.write("map:{}\n".format(self.env.terrain.name))
        file_desc.write("trans:{}\n".format(self.experience.total_trans))
        file_desc.write("use_a*_policy:{}\n".format(not self.random_policy))

    def load_experience(self, file, lock=None):
        self.experience = load_experience(file, lock)



