import copy
import os
import time
from collections import defaultdict

import numpy as np
import random as randomlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from random import random
from gym import Env
from gym.spaces import Discrete

from rl.agents.Agent import Agent
from rl.utils.networks.pd_network import NetApproximator
from rl.utils.functions import get_dict, set_dict, back_specified_dimension, process_experience_data, print_train_string
from rl.utils.policys import epsilon_greedy_policy, greedy_policy, uniform_random_policy, deep_epsilon_greedy_policy
from rl.utils.classes import SaveDictMixin, SaveNetworkMixin, Transition, Experience
from rl.utils.updates import soft_update

class QAgent(Agent,SaveDictMixin):
    def __init__(self,env:Env,capacity:int = 20000):
        super(QAgent, self).__init__(env,capacity)
        self.name = "QAgent"
        self.Q = {}

    def policy(self,A ,s = None,Q = None, epsilon = None):
        return epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self,lambda_ = None,gamma = 0.9,alpha = 0.1,
                        epsilon = 1e-5,display = False,wait = False,waitSecond:float = 0.01):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            #行动部分
            self.policy = epsilon_greedy_policy
            a0 = self.perform_policy(s0,self.Q, epsilon)
            s1, r1, is_done, info ,total_reward = self.act(a0)
            if display:
                self.env.render()
            #估值部分
            self.policy = greedy_policy
            a1 = greedy_policy(self.A, s1, self.Q)
            old_q = get_dict(self.Q,s0,a0)
            q_prime = get_dict(self.Q, s1, a1) #得到下一个状态，行为的估值
            td_target = r1 + gamma * q_prime
            new_q = old_q + alpha * (td_target - old_q)
            set_dict(self.Q, new_q, s0, a0)

            s0 = s1
            time_in_episode += 1
            if wait:
                time.sleep(waitSecond)

        print(self.experience.last_episode)
        return time_in_episode,total_reward

    def play_init(self,savePath,s0):
        self.Q = self.load_obj(savePath)
        return int(greedy_policy(self.A,s0,self.Q))

    def play_step(self,savePath,s0):
        return int(greedy_policy(self.A,s0,self.Q))

class DQNAgent(Agent,SaveNetworkMixin):
    def __init__(self,env:Env = None,
                    capacity = 20000,
                    hidden_dim:int = 32,
                    batch_size = 128,
                    epochs = 2,
                    tau:float = 0.1,
                    update_frequent = 50,
                    network:nn.Module = None):
        if env is None:
            raise Exception("agent should have an environment!")
        super(DQNAgent, self).__init__(env,capacity)
        self.name = "DQNAgent"
        self.input_dim = back_specified_dimension(env.observation_space)
        if type(env.action_space) is Discrete:
            self.output_dim = env.action_space.n
        else:
            raise Exception("DQN只能处理动作空间为Discrete的智能体!")
        self.hidden_dim = hidden_dim
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        # 行为网络，该网络用来计算产生行为，以及对应的Q值，参数频繁更新
        self.behavior_Q = network(input_dim=self.input_dim, output_dim=self.output_dim) \
                                    if network else NetApproximator(input_dim=self.input_dim,
                                                                    output_dim=self.output_dim,
                                                                    hidden_dim = self.hidden_dim)
        self.behavior_Q.to(self.device)
        #计算目标价值的网络，两者在初始时参数一致，该网络参数不定期更新
        self.target_Q = self.behavior_Q.clone()

        self.batch_size = batch_size
        self.epochs = epochs
        self.tau = tau
        self.update_frequent = update_frequent


    def _update_target_Q(self):
        # 使用软更新策略来缓解不稳定的问题
        soft_update(self.target_Q, self.behavior_Q, self.tau)

    def policy(self,A ,s = None,Q = None, epsilon = None):
        return deep_epsilon_greedy_policy(s, epsilon, self.env, self.behavior_Q)

    def learning_method(self,epsilon = 1e-5,display = False,wait = False,waitSecond:float = 0.01):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0
        while not is_done:
            a0 = self.perform_policy(s0,epsilon=epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            if self.total_trans > self.batch_size and time_in_episode % self.update_frequent == 0:
                loss += self._learn_from_memory(self.gamma,self.alpha)
            time_in_episode += 1

            s0 = s1

        loss /= time_in_episode
        if self.total_episodes_in_train % 500 == 0:
            print_train_string(self.experience)
        return time_in_episode, total_reward, loss

    def _learn_from_memory(self, gamma, learning_rate):
        trans_pieces = self.sample(self.batch_size)
        states_0, actions_0, \
        reward_1, is_done, states_1 = process_experience_data(trans_pieces)

        #准备训练数据
        X_batch = states_0
        y_batch = self.target_Q(states_0)
        Q_target = reward_1 + gamma * np.max(self.target_Q(states_1),axis=1)*\
                   (~is_done) # is_done则Q_target==reward_1

        y_batch[np.arange(len(X_batch)), actions_0] = Q_target

        X_batch = torch.from_numpy(X_batch)
        y_batch = torch.from_numpy(y_batch)

        #训练行为价值网络，更新其参数
        loss = self.behavior_Q.fit(x = X_batch,
                                   y = y_batch,
                                   learning_rate = learning_rate,
                                   epochs=self.epochs)
        mean_loss = loss.sum().item() / self.batch_size
        if self.total_episodes_in_train % 100 == 0:
            self._update_target_Q()
        return mean_loss

    def play_init(self,savePath,s0):
        self.load(os.path.join(savePath,"DQNAgent.pkl"),self.behavior_Q)
        return int(np.argmax(self.behavior_Q(s0)))

    def play_step(self,savePath,s0):
        return int(np.argmax(self.behavior_Q(s0)))

class Deep_DYNA_QAgent(Agent, SaveNetworkMixin):
    def __init__(self, env:Env, hidden_dim=32,
                 batch_size:int=128,
                 tau: float = 0.1,
                 update_frequent = 50,
                 capacity:int = 20000,
                 plan_capacity:int = 10000,
                 lookahead:int=10,
                 epochs:int=2,
                 learning_rate:float = 1e-4,
                 Q_net:nn.Module = None,
                 model_net:nn.Module = None):
        super(Deep_DYNA_QAgent, self).__init__(env, capacity)
        self.name = "Deep_DYNA_QAgent"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.input_dim = back_specified_dimension(env.observation_space)
        if type(env.action_space) is Discrete:
            self.output_dim = env.action_space.n
        else:
            raise Exception("DDQ只能处理动作空间为Discrete的智能体!")

        self.Q = Q_net if Q_net else NetApproximator(self.input_dim, self.output_dim, hidden_dim)
        self.target_Q = copy.deepcopy(self.Q)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), learning_rate)

        self.model = model_net()
        self.model_optimizer = torch.optim.Adam(self.Q.parameters(), learning_rate)

        self.lookahead = lookahead
        self.batch_size = batch_size
        self.tau = tau
        self.epochs = epochs
        self.update_frequent = update_frequent

        self.plan_experience = Experience(capacity=plan_capacity)


    def policy(self,A ,s = None,Q = None, epsilon = None):
        return deep_epsilon_greedy_policy(s, epsilon, self.env, self.Q)

    def learning_method(self,lambda_ = None,gamma = 0.9,alpha = 0.1,
                        epsilon = 1e-5,display = False,wait = False,waitSecond:float = 0.01):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0.0

        while not is_done:
            #行动部分
            self.policy = deep_epsilon_greedy_policy
            a0 = self.perform_policy(s0, self.Q, epsilon)
            s1, r1, is_done, info , total_reward = self.act(a0)
            if display:
                self.env.render()
            if wait:
                time.sleep(waitSecond)
            time_in_episode += 1
            if self.total_trans > self.batch_size and time_in_episode % self.update_frequent == 0:
                loss += self._learn_from_memory(self.gamma, self._sample_from_du)
                self._learn_simulate_world()
                self._planning(self.gamma)
            s0 = s1
        if self.total_episodes_in_train % 500 == 0:
            print_train_string(self.experience)
        loss /= time_in_episode

        return time_in_episode,total_reward, loss

    def _update_target_Q(self):
        # 使用软更新策略来缓解不稳定的问题
        soft_update(self.target_Q, self.Q, self.tau)

    def _sample_from_du(self, count=0):
        '''
        从经验中抽取值
        :return:
        '''
        return self.sample(self.batch_size if count == 0 else count)

    def _sample_from_ds(self, count=0):
        '''
        从planning经验中抽取值
        :return:
        '''
        return self.plan_experience.sample(self.batch_size if count == 0 else count)

    def _learn_from_memory(self, gamma, sample_func):
        trans_pieces = sample_func(self.batch_size)
        states_0, actions_0, \
        reward_1, is_done, states_1 = process_experience_data(trans_pieces)

        # 准备训练数据
        X_batch = states_0
        y_batch = self.target_Q(states_0)
        Q_target = reward_1 + gamma * np.max(self.target_Q(states_1), axis=1) * \
                   (~is_done)  # is_done则Q_target==reward_1

        y_batch[np.arange(len(X_batch)), actions_0] = Q_target

        X_batch = torch.from_numpy(X_batch)
        y_batch = torch.from_numpy(y_batch)

        # 训练行为价值网络，更新其参数
        loss = F.mse_loss(X_batch, y_batch)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

        mean_loss = loss.sum().item() / self.batch_size
        # will update later!
        # if self.experience.total_trans % 100 == 0:
        #     self._update_target_Q()
        return mean_loss

    def _learn_simulate_world(self):
        trans_pieces = self.sample(self.batch_size)
        states_0, actions_0, \
        reward_1, is_done, states_1 = process_experience_data(trans_pieces)

        states_1 = torch.from_numpy(states_1)
        reward_1 = torch.from_numpy(reward_1)

        #world model输入为(s,a)，输出为(s',r, is_done)
        _state_1, _reward, _is_done = self.model(states_0, actions_0)

        self.model_optimizer.zero_grad()
        loss = F.mse_loss(_state_1, states_1) + \
                F.mse_loss(_reward, reward_1) + \
                F.binary_cross_entropy_with_logits(_is_done, is_done)
        loss.backward()
        self.model_optimizer.step()

        mean_loss = loss.sum().item() / self.batch_size
        return mean_loss

    def _planning(self, gamma):
        trans_piece = self._sample_from_du(self.lookahead)
        states_0, actions_0, _, _, _ = process_experience_data(trans_piece)

        states_0 = torch.from_numpy(states_0)
        actions_0 = torch.from_numpy(actions_0)
        rewards, states_1, is_done = self.model(states_0, actions_0)

        for i in range(len(states_0)):
            trans = Transition(states_0[i], actions_0[i], rewards[i], is_done[i], states_1[i])
            self.plan_experience.push(trans)
        #从planning经验中开始学习
        if self.plan_experience.len >= self.batch_size:
            self._learn_from_memory(gamma, self._sample_from_ds)

    def play_init(self,savePath,s0):
        self.load(savePath,self.Q)
        return int(np.argmax(self.Q(s0)))

    def play_step(self,savePath,s0):
        return int(np.argmax(self.Q(s0)))