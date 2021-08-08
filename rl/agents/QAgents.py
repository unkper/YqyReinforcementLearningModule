import time
import numpy as np
import random as randomlib
import torch
import torch.nn as nn

from random import random
from gym import Env
from gym.spaces import Discrete

from rl.agents.Agent import Agent
from rl.utils.networks import NetApproximator
from rl.utils.functions import get_dict, set_dict, back_specified_dimension
from rl.utils.policys import epsilon_greedy_policy, greedy_policy, uniform_random_policy
from rl.utils.classes import SaveDictMixin, SaveNetworkMixin, SimulationEnvModel, Transition
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
        if network is not None:
            self.behavior_Q = network(input_dim=self.input_dim,
                                      output_dim=self.output_dim)
        else:
            # 行为网络，该网络用来计算产生行为，以及对应的Q值，参数频繁更新
            self.behavior_Q = NetApproximator(input_dim=self.input_dim,
                                              output_dim=self.output_dim,
                                              hidden_dim = self.hidden_dim)
        self.behavior_Q.to(self.device)
        #计算目标价值的网络，两者在初始时参数一致，该网络参数不定期更新
        self.target_Q = self.behavior_Q.clone()

        self.batch_size = batch_size
        self.epochs = epochs
        self.tau = tau

    def _update_target_Q(self):
        # 使用软更新策略来缓解不稳定的问题
        soft_update(self.target_Q, self.behavior_Q, self.tau)

    def policy(self,A ,s = None,Q = None, epsilon = None):
        action = self.behavior_Q(s) #经过神经网络输出一个[1,5]向量，里面代表前，后，左，右，不动的值，之后选取其中最大的输出
        rand_value = random()
        if epsilon is not None and rand_value < epsilon:
            return self.env.action_space.sample() #有ε概率随机选取动作
        else:
            return int(np.argmax(action))

    def learning_method(self,lambda_ = None,gamma = 0.9,alpha = 0.1,
                        epsilon = 1e-5,display = False,wait = False,waitSecond:float = 0.01):
        self.state = self.env.reset()
        s0 = self.state #当前状态特征
        if display:
            self.env.render()
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0
        while not is_done:
            s0 = self.state
            a0 = self.perform_policy(s0,epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            if self.total_trans > self.batch_size:
                loss += self._learn_from_memory(gamma,alpha)
            time_in_episode += 1

        loss /= time_in_episode

        print("epsilon:{:3.2f},loss:{:3.2f},{}".format(epsilon, loss, self.
                                                           experience.last_episode))
        return time_in_episode, total_reward

    def _learn_from_memory(self, gamma, learning_rate):
        trans_pieces = self.sample(self.batch_size)
        states_0 = np.vstack([x.s0 for x in trans_pieces])
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack(x.s1 for x in trans_pieces)

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
        self._update_target_Q()
        return mean_loss

    def play_init(self,savePath,s0):
        self.load(savePath,self.behavior_Q)
        return int(np.argmax(self.behavior_Q(s0)))

    def play_step(self,savePath,s0):
        return int(np.argmax(self.behavior_Q(s0)))

class DYNA_QAgent(Agent, SaveDictMixin):
    def __init__(self,env:Env=None,capacity:int = 20000,learn_sim_model_count:int=10):
        super(DYNA_QAgent, self).__init__(env, capacity)
        self.name = "Tabular_DYNA_QAgent"
        self.Q = {}
        self.env_model = SimulationEnvModel()
        self.learn_sim_model_count = learn_sim_model_count

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
        learn_sim_model_limit = self.env_model.maxSize * 3 / 4
        while not is_done:
            #行动部分
            self.policy = epsilon_greedy_policy
            a0 = self.perform_policy(s0,self.Q, epsilon)
            s1, r1, is_done, info ,total_reward = self.act(a0)
            if display:
                self.env.render()
            #估值部分
            self.policy = greedy_policy
            a1 = self.perform_policy(s1,self.Q,epsilon)
            old_q = get_dict(self.Q, s0, a0)
            q_prime = get_dict(self.Q, s1, a1) #得到下一个状态，行为的估值
            td_target = r1 + gamma * q_prime
            new_q = old_q + alpha * (td_target - old_q)
            set_dict(self.Q, new_q, s0, a0)
            #存储当前s0,a0,r,s1到model中去
            self.env_model.push(s0,a0,r1,s1,is_done)
            s0 = s1
            time_in_episode += 1
            if len(self.env_model) > learn_sim_model_limit:
                for n in range(self.learn_sim_model_count):
                    s,a = randomlib.sample(self.env_model.keys(),k=1)[0]
                    r,s1,is_done = self.env_model[(s,a)]
                    old_q = get_dict(self.Q, s0, a0)
                    if is_done:
                        td_target = r
                    else:
                        self.policy = greedy_policy
                        td_target = r + gamma * get_dict(self.Q, s1, self.perform_policy(s1,self.Q,epsilon))
                    set_dict(self.Q, old_q * alpha * td_target, s0, a0)
            if wait:
                time.sleep(waitSecond)

        print(self.experience.last_episode)
        return time_in_episode,total_reward

    def play_init(self,savePath,s0):
        self.Q = self.load_obj(savePath)
        return int(greedy_policy(self.A,s0,self.Q))

    def play_step(self,savePath,s0):
        return int(greedy_policy(self.A,s0,self.Q))