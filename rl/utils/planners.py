import copy
import datetime
import os
import pickle
import time

import numpy as np
from tqdm import tqdm

from ped_env.pathfinder import AStarController
from rl.agents.Agent import Agent
from rl.utils.classes import Experience, Transition, make_parallel_env

def init_offline_train(agent:Agent, data, episode):
    print("开始进行离线BC训练...")
    planner = AStarPlanner(agent.env, load_mode=True)
    planner.load_experience(data)
    for i in tqdm(range(episode)):
        trans_pieces = planner.experience.sample(agent.batch_size)
        loss_c, loss_a = agent._learn_from_memory(trans_pieces, True)
        agent.writer.add_scalar("init/loss_critic", loss_c, i)
        agent.writer.add_scalar("init/loss_actor", loss_a, i)
    sname = agent.log_dir
    print("save network!......")
    for i in range(agent.env.agent_count):
        agent.save(sname, "Actor{}".format(i), agent.agents[i].actor, 0)
        agent.save(sname, "Critic{}".format(i), agent.agents[i].critic, 0)
    print("将演示经验赋值给智能体!")
    agent.demo_experience = planner.experience

class AStarPlanner:
    def __init__(self, env, capacity=1e6, n_rol_threads=1, load_mode=False, use_random_policy=False):
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
            self.controler = AStarController(env, use_random_policy) if n_rol_threads == 1 else make_parallel_env(AStarController(env, use_random_policy), n_rol_threads)

    def planning(self, episodes=1):
        for epoch in tqdm(range(0, episodes, self.n_rol_threads)):
            step, starttime = 0, time.time()
            obs = self.controler.reset()
            is_done = np.array([False])
            while not is_done.any():
                next_obs, reward, is_done, action = self.controler.step(obs)
                if self.n_rol_threads == 1:
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
            print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime, step / (endtime - starttime)))

    def clear_experience(self):
        self.experience.clear()

    def save_experience(self):
        dir_name = "{}_exp".format(self.env.terrain.name)
        if self.random_policy:
            dir_name += "_random"
        os.mkdir("./data/exp/" + dir_name)
        file_exp = open(os.path.join("./data/exp", dir_name,
                                     "experience.pkl"),"wb")
        pickle.dump(self.experience,file_exp,0)
        file_desc = open(os.path.join("./data/exp", dir_name,
                                     "desc.txt"), "w+")
        file_desc.write("person_num:{}\n".format(self.env.person_num))
        file_desc.write("group_size:{}\n".format(self.env.group_size))
        file_desc.write("random_init:{}\n".format(self.env.random_init_mode))
        file_desc.write("map:{}\n".format(self.env.terrain.name))
        file_desc.write("trans:{}\n".format(self.experience.total_trans))
        file_desc.write("use_a*_policy:{}\n".format(not self.random_policy))

    def load_experience(self, file):
        file = open(file,"rb")
        self.experience = pickle.load(file)


