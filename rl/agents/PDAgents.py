import random
import time

import torch
import torch.nn.functional as F
import numpy as np

from gym import Env
from gym.spaces import Discrete
from torch import nn
from torch.autograd import Variable

from rl.agents.Agent import Agent
from rl.utils.networks.pd_network import Critic, Actor, SimpleCritic, SimpleActor02
from rl.utils.updates import soft_update, hard_update
from rl.utils.classes import SaveNetworkMixin,Noise
from rl.utils.functions import back_specified_dimension, onehot_from_int


class DDPGAgent(Agent,SaveNetworkMixin):

    def __init__(self,env:Env = None,
                      capacity = 2e6,
                      batch_size = 128,
                      learning_rate = 0.001,
                      epochs = 2,
                      actor_network:nn.Module=None,
                      critic_network:nn.Module=None):
        if env is None:
            raise Exception("agent should have an environment!")
        super(DDPGAgent, self).__init__(env,capacity)
        self.state_dim = back_specified_dimension(env.observation_space)
        self.action_dim = back_specified_dimension(env.action_space)
        if type(env.action_space) is Discrete:
            self.discrete = True
        else:
            self.discrete = False
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = 0.999
        self.epochs = epochs
        self.tau = 0.001
        self.noise = Noise(self.action_dim)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.actor = SimpleActor02(self.state_dim,
                                   self.action_dim,
                                   self.discrete).to(self.device) if actor_network == None else actor_network(self.state_dim, self.action_dim)
        self.target_actor = SimpleActor02(self.state_dim,
                                          self.action_dim,
                                          self.discrete).to(self.device) if actor_network == None else actor_network(self.state_dim, self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                self.learning_rate)
        self.critic = SimpleCritic(self.state_dim, self.action_dim).to(self.device) if critic_network == None else critic_network(self.state_dim, self.action_dim)
        self.target_critic = SimpleCritic(self.state_dim, self.action_dim).to(self.device) if critic_network == None else critic_network(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 self.learning_rate)
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.rewards = []

        return

    def get_exploitation_action(self,state):
        '''
        得到给定状态下依据目标演员网络计算出的行为，不探索
        :param state: numpy数组
        :return: 动作 numpy数组
        '''
        action = self.actor.forward(state).detach()
        return action

    def get_exploration_action(self, state, epsilon):
        '''
        得到给定状态下根据演员网络计算出的带噪声的行为，模拟一定的探索
        :param state: numpy数组
        :return: action numpy数组
        '''
        action = self.actor.forward(state).detach()
        value = random.random()
        if self.discrete:
            if value < epsilon:
                action = onehot_from_int(random.sample(range(0,self.action_dim),1), self.action_dim).to(self.device)
        else:
            if value < epsilon:
                action += torch.tensor(self.noise.sample()).to(self.device)
            action = action.clamp(-1,1)
        return action

    def play_init(self,savePath,s0):
        self.load(savePath,self.target_actor)
        return self.get_exploitation_action(s0)

    def play_step(self,savePath,s0):
        return self.get_exploitation_action(s0)

    def _learn_from_memory(self):
        '''
        从记忆学习，更新两个网络的参数
        :return:
        '''
        # 随机获取记忆里的Transmition
        trans_pieces = self.sample(self.batch_size)
        s0 = np.vstack([x.s0 for x in trans_pieces])
        a0 = np.array([x.a0 for x in trans_pieces])
        r1 = np.array([x.reward for x in trans_pieces])
        # is_done = np.array([x.is_done for x in trans_pieces])
        s1 = np.vstack([x.s1 for x in trans_pieces])

        a0 = torch.from_numpy(a0).to(self.device)
        s0 = torch.from_numpy(s0).to(self.device)
        #优化评价家网络参数
        s1 = torch.from_numpy(s1).to(self.device)
        a1 = self.target_actor.forward(s1).detach()
        next_val = torch.squeeze(self.target_critic.forward(s1,a1)).detach() #必须detach!
        # y_exp = r + gamma * Q'( s', pi'(s'))
        y_expected = torch.from_numpy(r1).to(self.device) + self.gamma * next_val
        y_expected = y_expected.type(torch.FloatTensor).to(self.device)
        # y_pred = Q( s1, a1)

        y_predicted = torch.squeeze(self.critic.forward(s0,a0))
        loss_critic = F.smooth_l1_loss(y_predicted,y_expected).to(self.device)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        #优化演员网络参数，优化的目标是使得Q增大
        pred_a0 = self.actor.forward(s0)
        #反向梯度下降，
        loss_actor = -1 * self.critic.forward(s0,pred_a0).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        #软更新参数
        if self.total_trans_in_train % 100 == 0:
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)
        return (loss_critic.item(), loss_actor.item())

    def learning_method(self,
                        epsilon = 0.2,explore = True,display = False,
                        wait = False,waitSecond:float = 0.01):
        self.state = np.float64(self.env.reset())
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss_critic, loss_actor = 0.0,0.0
        s0 = self.state
        while not is_done:
            if explore:
                a0 = self.get_exploration_action(s0,epsilon)
            else:
                a0 = self.actor.forward(s0).detach().data.numpy()
            s1,r1,is_done,info,reward = self.act(a0)
            if display:
                self.env.render()
            if self.total_trans > self.batch_size and self.total_trans_in_train % 50 == 0:
                loss_c, loss_a = self._learn_from_memory()
                loss_critic += loss_c
                loss_actor += loss_a
            s0 = s1
            time_in_episode += 1
            self.total_trans_in_train += 1
            total_reward += reward
            if wait:
                time.sleep(waitSecond)

        loss_critic /= time_in_episode
        loss_actor /= time_in_episode
        loss = loss_critic + loss_actor

        self.rewards.append(total_reward)
        if self.total_trans_in_train % 500 == 0:
            print("{}".format(self.experience.__str__()))
            print("average reward in last 500 episodes:{}".format(np.mean(self.rewards).item()))
            self.rewards = []
        return time_in_episode,total_reward, loss

    # def save_models(self, episode_count):
    #     torch.save(self.target_actor.state_dict(),'./Models/'+str(
    #         episode_count
    #     ) + '_actor.pt')
    #     torch.save(self.target_critic.state_dict(),'./Models/'+str(
    #         episode_count
    #     )+ '_critic.pt')
    #     print("Model saved successfully!")
    #
    # def load_models(self, episode):
    #     self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
    #     self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
    #     hard_update(self.target_actor, self.actor)
    #     hard_update(self.target_critic, self.critic)
    #     print("Models loaded succesfully")