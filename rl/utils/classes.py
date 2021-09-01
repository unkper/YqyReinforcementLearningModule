import datetime
import os
import pickle
import math
from queue import Queue

import gym
import random
import numpy as np
import torch

from typing import List
from torch import nn

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
        return curEpisode.push(trans)

    def sample(self, batch_size=1): # sample transition
        '''randomly sample some transitions from agent's experience.abs
        随机获取一定数量的状态转化对象Transition
        args:
            number of transitions need to be sampled
        return:
            list of Transition.
        '''
        sample_trans = []
        while batch_size > 0:
            index = int(random.random() * self.len)
            episode_len = self.episodes[index].len
            count = int(round(random.random() * episode_len))
            count = min(count, batch_size, episode_len)
            sample_trans += self.episodes[index].sample(count)
            batch_size -= count
        return sample_trans

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

    def save(self,save_file_name:str,name:str,network:nn.Module):
        p = os.path.join("./",save_file_name)
        if not os.path.exists(p):
            os.mkdir(p)
        save_name = os.path.join("./",save_file_name,"./{}.pkl".format(name))
        torch.save(network.state_dict(),save_name)
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

class SimulationEnvModel():
    def __init__(self,maxLength=2000):
        self.maxSize = maxLength
        self.queue = Queue(maxsize=-1)
        self.dic = {}

    def push(self,s,a,r,s1,is_done):
        if self.queue.qsize() + 1 > self.maxSize:
            ele = self.queue.get()
            del self.dic[ele]
        key = [s, a]
        if key in self.dic.keys():
            pass
        else:
            self.dic[key] = (r, s1, is_done)
            self.queue.put(key)

    def keys(self):
        return self.dic.keys()

    def __getitem__(self, item):
        return self.dic[item]

    def __len__(self):
        return len(self.dic)

