import time

from gym import Env

from .Agent import Agent
from rl.utils.functions import get_dict, set_dict
from rl.utils.policys import epsilon_greedy_policy, greedy_policy
from rl.utils.classes import SaveDictMixin

class SarsaAgent(Agent,SaveDictMixin):
    def __init__(self,env:Env,capacity:int = 20000):
        super(SarsaAgent, self).__init__(env,capacity)
        self.Q = {} #增加Q字典存储行为价值

    def policy(self,A ,s = None,Q = None, epsilon = None):
        '''
        使用epsilon-贪婪策略
        :param A:
        :param s:
        :param Q:
        :param epsilon:
        :return:
        '''
        return epsilon_greedy_policy(A,s,Q,epsilon)

    def learning_method(self,lambda_ = 0.9,gamma = 0.9,alpha = 0.5,
                        epsilon = 0.2,display = False,wait = False,waitSecond:float = 0.01):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0,self.Q,epsilon)
        time_in_episode, total_reward = 0,0
        is_done = False
        while not is_done:
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1,self.Q,epsilon)
            old_q = get_dict(self.Q,s0,a0)
            q_prime = get_dict(self.Q,s1,a1)
            td_target = r1 + gamma * q_prime
            new_q = old_q + alpha * (td_target - old_q)
            set_dict(self.Q,new_q,s0,a0)
            s0,a0 = s1,a1
            time_in_episode += 1
            if wait:
                time.sleep(waitSecond)
        if display:
            print(self.experience.last_episode)
        return time_in_episode,total_reward

    def play_init(self,savePath,s0):
        self.Q = self.load_obj(savePath)
        return int(greedy_policy(self.A,s0,self.Q))

    def play_step(self,savePath,s0):
        return int(greedy_policy(self.A,s0,self.Q))

class SarsaLambdaAgent(Agent,SaveDictMixin):
    def __init__(self,env:Env,capacity:int = 20000):
        super(SarsaLambdaAgent, self).__init__(env,capacity)
        self.Q = {}

    def policy(self,A,s,Q,epsilon):
        return epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self,lambda_ = 0.9,gamma = 0.9,alpha = 0.1,
                        epsilon = 1e-5,display = False,wait = False,waitSecond:float = 0.01):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0,self.Q,epsilon)
        time_in_episode, total_reward = 0,0
        is_done = False
        E = {} #效用值
        while not is_done:
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1,self.Q,epsilon)
            q = get_dict(self.Q, s0, a0)
            q_prime = get_dict(self.Q,s1,a1)
            delta = r1 + gamma * q_prime - q
            e = get_dict(E,s0,a0)
            e += 1
            set_dict(E,e,s0,a0)
            for s in self.S: #对所有可能的Q(s,a)进行更新
                for a in self.A:
                    e_value = get_dict(E,s,a)
                    old_q = get_dict(self.Q,s,a)
                    new_q = old_q + alpha * delta * e_value
                    new_e = gamma * lambda_ * e_value
                    set_dict(self.Q,new_q,s,a)
                    set_dict(E,new_e,s,a)

            s0, a0 = s1, a1
            time_in_episode += 1
            if wait:
                time.sleep(waitSecond)
        if display:
            print(self.experience.last_episode)
        return time_in_episode,total_reward

    def play_init(self,savePath,s0):
        self.Q = self.load_obj(savePath)
        return int(greedy_policy(self.A,s0,self.Q))

    def play_step(self,savePath,s0):
        return int(greedy_policy(self.A,s0,self.Q))