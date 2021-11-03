import datetime
import os
import pickle
import time

import gym
import random
import numpy as np
import torch

from typing import List
from torch import nn

from ped_env.envs import PedsMoveEnv
from rl.utils.functions import flatten_data
from rl.utils.updates import hard_update


class State():
    def __init__(self, name):
        self.name = name


class Transition():

    def __init__(self, s0, a0, reward: float, is_done: bool, s1):
        self.data = [s0, a0, reward, is_done, s1]

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        return "s0:{};a0:{};reward:{:.3f};is_done:{};s1:{};\n".format(
            self.data[0], self.data[1], self.data[2], self.data[3],
            self.data[4], self.data[5]
        )

    @property
    def s0(self):
        return self.data[0]

    @property
    def a0(self):
        return self.data[1]

    @property
    def reward(self):
        return self.data[2]

    @property
    def is_done(self) -> bool:
        return self.data[3]

    @property
    def s1(self):
        return self.data[4]

class Episode():

    def __init__(self, id: int = 0) -> None:
        self.total_reward = 0  # 总的奖励值
        self.trans_list = []  # 状态转移序列
        self.name = str(id)  # 名称

    def push(self, trans: Transition) -> List[float]:
        '''
        将一个状态转换送入状态序列中，返回该序列当前总的奖励值(不计衰减)
        :param trans:
        :return:
        '''
        self.trans_list.append(trans)
        if type(trans.reward) is list:
            if type(self.total_reward) is not list:self.total_reward = [0.0 for _ in range(len(trans.reward))]
            for idx, value in enumerate(trans.reward):
                self.total_reward[idx] += value
        else:
            self.total_reward += trans.reward  # 不计衰减的总奖励
        return self.total_reward

    @property
    def len(self):
        return len(self.trans_list)

    def __str__(self):
        if isinstance(self.total_reward,list):
            return "episode {}:{} steps,total reward:{}\n". \
                    format(self.name, self.len, self.total_reward)
        return "episode {0:<4}:{1:>4} steps,total reward:{2:<8.2f}\n". \
            format(self.name, self.len, self.total_reward)

    def is_compute(self) -> bool:
        if self.len == 0:
            return False
        is_done = self.trans_list[self.len - 1].is_done
        if type(is_done) is list:return is_done[0]
        return self.trans_list[self.len - 1].is_done

    def pop(self) -> Transition:
        if self.len == 0: return None
        var = self.trans_list.pop()
        if type(var.reward) is list:
            if type(self.total_reward) is not list:raise Exception()
            for idx, value in enumerate(var.reward):
                self.total_reward[idx] += value
        else:
            self.total_reward -= var.reward
        return var

    def sample(self,batch_size = 1)->list:
        return random.sample(self.trans_list,batch_size)

    def __len__(self):
        return self.len

class Experience():
    '''
    该类是用来存储智能体的相关经历的，它由一个列表所组成，
    该类可以通过调用方法来随机返回几个不相关的序列或是经历
    '''
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity  # 容量：指的是trans总数量
        self.episodes = []  # episode列表
        self.transitions = []
        self.next_id = 0  # 下一个episode的Id
        self.total_trans = 0  # 总的状态转换数量

    def __str__(self):
        return "exp info:{0:5} episodes, memory usage {1}/{2}". \
            format(self.len, self.total_trans, self.capacity)

    @property
    def len(self):
        return len(self.episodes)

    def __len__(self):
        return self.len

    def _remove(self,index = 0) -> Episode:
        '''
        丢弃一个episode，默认第一个
        :param index: 要丢弃的episode的编号
        :return: 丢弃的episode
        '''
        if index > self.len - 1:
            raise Exception("Invaild Index!!!")
        if self.len > 0:
            episode = self.episodes[index]
            self.episodes.remove(episode)
            trans_set = set(episode.trans_list)
            self.transitions = [x for x in self.transitions if x not in trans_set]
            self.total_trans -= episode.len
            return episode
        else:
            return None

    def _remove_first(self):
        self._remove(0)

    def push(self, trans:Transition):
        if self.capacity <= 0:
            return
        while self.capacity <= self.total_trans:
            self._remove_first()
        self.total_trans += 1
        curEpisode = None
        if self.len == 0 or self.episodes[self.len - 1].is_compute() == True:
            curEpisode = Episode(self.next_id)
            self.next_id += 1
            self.episodes.append(curEpisode)
        else:
            curEpisode = self.episodes[self.len-1]
        self.transitions.append(trans)
        return curEpisode.push(trans)

    def sample(self, batch_size=1): # sample transition
        '''randomly sample some transitions from agent's experience.abs
        随机获取一定数量的状态转化对象Transition
        args:
            number of transitions need to be sampled
        return:
            list of Transition.
        '''
        return random.sample(self.transitions, batch_size)

    def sample_episode(self, episode_num = 1):  # sample episode
        '''随机获取一定数量完整的Episode
        '''
        return random.sample(self.episodes, k = episode_num)

    def last_n_episode(self,N):
        if self.len >= N:
            return self.episodes[self.len - N : self.len]
        return None

    @property
    def last_episode(self):
        if self.len > 0:
            return self.episodes[self.len-1]
        return None

class OrnsteinUhlenbeckActionNoise():
    '''
    用于连续动作空间的噪声辅助类，输出具有扰动的一系列值
    '''
    def __init__(self, action_dim, mu = 0,theta = 0.15, sigma = 0.2):
        '''
        动作
        :param action_dim:动作空间的维数
        :param mu:
        :param theta:
        :param sigma:
        '''
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

class SaveNetworkMixin():

    def save(self,sname:str,name:str,network:nn.Module):
        p = os.path.join("./",sname)
        if not os.path.exists(p):
            os.mkdir(p)
        save_name = os.path.join("./",sname,"./{}.pkl".format(name))
        torch.save(network.state_dict(),save_name)
        desc_txt_file = open(os.path.join(sname, "desc.txt"),"w+")
        desc_txt_file.write("algorithm:" + str(self) + "\n")
        desc_txt_file.write("batch_size:" + str(self.batch_size) + "\n")
        desc_txt_file.write("update_freq:" + str(self.update_frequent) + "\n")
        desc_txt_file.write("lr:" + str(self.learning_rate) + "\n")
        desc_txt_file.write("gamma:" + str(self.gamma) + "\n")
        desc_txt_file.write("envName:" + str(self.env_name) + "\n")
        desc_txt_file.write("agent_count:" + str(self.env.agent_count) + "\n")
        desc_txt_file.write("hidden_dim:" + str(self.hidden_dim) + "\n")
        if isinstance(self.env, PedsMoveEnv):
            desc_txt_file.write("person_num:" + str(self.env.person_num) + "\n")
            desc_txt_file.write("group_size:" + str(self.env.group_size) + "\n")
        return save_name

    def load(self,savePath,network:nn.Module):
        network.load_state_dict(torch.load(savePath))

class SaveDictMixin():
    def save_obj(self,obj, name):
        save_name = os.path.join("./",name+'.pkl')
        with open(save_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return save_name

    def load_obj(self,savePath):
        with open(savePath , 'rb') as f:
            return pickle.load(f)

class MAAgentMixin():
    def get_exploitation_action(self, state):
        """
        得到给定状态下依据目标演员网络计算出的行为，不探索
        :param state: numpy数组
        :return: 动作 numpy数组
        """
        action_list = []
        for i in range(self.env.agent_count):
            s = flatten_data(state[i], self.state_dims[i], self.device)
            action = self.agents[i].step(s, False).detach().cpu().numpy()
            action_list.append(action)
        action_list = np.array(action_list,dtype=object)
        return action_list

    def get_exploration_action(self, state, epsilon=0.1):
        '''
        得到给定状态下根据演员网络计算出的带噪声的行为，模拟一定的探索
        :param state: numpy数组
        :return: action numpy数组
        '''
        action_list = []
        value = random.random()
        for i in range(self.env.agent_count):
            s = flatten_data(state[i], self.state_dims[i], self.device)
            action = self.agents[i].step(s, True if value < epsilon else False).detach().cpu().numpy()
            action_list.append(action)
        action_list = np.array(action_list,dtype=object)
        return action_list

    def play_init(self, savePath, s0):
        import os
        for i in range(self.env.agent_count):
            saPath = os.path.join(savePath, "Actor{}.pkl".format(i))
            self.load(saPath, self.agents[i].actor)
            hard_update(self.agents[i].target_actor, self.agents[i].actor)
        return self.get_exploitation_action(s0)

    def play_step(self, savePath, s0):
        return self.get_exploitation_action(s0)

    def learning_method(self, epsilon=0.2, explore=True, display=False,
                        wait=False, waitSecond: float = 0.01):
        self.state = self.env.reset()
        time_in_episode, total_reward = 0, 0
        is_done = [False]
        loss_critic, loss_actor = 0.0, 0.0
        s0 = self.state
        #is_done此时已经为数组
        while not is_done[0]:
            if explore:
                a0 = self.get_exploration_action(s0, epsilon)
            else:
                a0 = self.get_exploitation_action(s0)
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            if self.total_trans > self.batch_size and self.total_trans_in_train % self.update_frequent == 0:
                loss_c, loss_a = self._learn_from_memory()
                loss_critic += loss_c
                loss_actor += loss_a
            time_in_episode += 1
            self.total_trans_in_train += 1
            s0 = s1
            if wait:
                time.sleep(waitSecond)

        loss_critic /= time_in_episode
        loss_actor /= time_in_episode

        if self.total_episodes_in_train > 0 \
                and self.total_episodes_in_train % self.log_frequent == 0:
            rewards = []
            last_episodes = self.experience.last_n_episode(self.log_frequent)
            for i in range(self.env.agent_count):
                rewards.append(np.mean([x.total_reward[i] for x in last_episodes]))
            print("average rewards in last {} episodes:{}".format(self.log_frequent, rewards))
            print("{}".format(self.experience.__str__()))
            for i, agent in enumerate(self.agents):
                print("Agent{}:{}".format(i, agent.count))
                agent.count = [0 for _ in range(agent.action_dim)]
        return time_in_episode, total_reward, [loss_critic, loss_actor]

if __name__ == "__main__":
    noise = OrnsteinUhlenbeckActionNoise(2)
    for i in range(1000):
        print(noise.sample())