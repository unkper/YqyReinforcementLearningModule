import random
import time

import torch
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from gym import Env
from gym.spaces import Discrete
from torch import nn

from rl.agents.Agent import Agent
from rl.utils.networks import SimpleActor, SimpleCritic
from rl.utils.updates import soft_update, hard_update
from rl.utils.classes import SaveNetworkMixin, OrnsteinUhlenbeckActionNoise, Experience, Transition
from rl.utils.functions import back_specified_dimension, onehot_from_int, gumbel_softmax, flatten_data


class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_lim,
                 learning_rate, discrete, capicity,
                 device,global_dim = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        self.actor = SimpleActor(state_dim, action_dim, action_lim).to(self.device)
        self.target_actor = SimpleActor(state_dim, action_dim, action_lim).to(self.device)
        hard_update(self.target_actor, self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                learning_rate)
        dim = global_dim if global_dim is not None else self.state_dim
        self.critic = SimpleCritic(dim, action_dim).to(self.device)
        self.target_critic = SimpleCritic(dim, action_dim).to(self.device)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 learning_rate)
        self.noise = OrnsteinUhlenbeckActionNoise(1 if self.discrete else action_dim)
        self.experience = Experience(capicity)

    def step(self, obs, explore):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (Pytorch Variable): Actions for this agent
        """
        action = self.actor(obs)
        if self.discrete:
            if explore:
                action = onehot_from_int(random.sample(range(0, self.action_dim), 1)[0]
                                                  , self.action_dim).to(self.device)
        else:
            if explore:
                action += torch.Tensor(self.noise.sample()).to(self.device)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_actor.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.actor.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_actor.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.actor_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class MADDPGAgent(Agent, SaveNetworkMixin):

    def __init__(self, env: Env = None,
                 capacity=2e6,
                 batch_size=128,
                 learning_rate=0.001,
                 use_global_state=True
                 ):
        """
        环境的输入有以下几点变化，设此时有N个智能体：
        状态为(o1,o2,...,oN)
        每个状态o的形状暂定为一样，对于Actor有如下几种情况：
            类型为Discrete，输入层为1，输出层为需要动作空间数
            类型为Box，其Shape为（x1,x2,...,xn)，则输入层为x1*x2*xn
        对于Critic
        动作一般为一维的Box，则根据维数来进行转换

        :param env:
        :param capacity:
        :param batch_size:
        :param action_lim:
        :param learning_rate:
        :param epochs:
        """
        if env is None:
            raise Exception("agent should have an environment!")
        super(MADDPGAgent, self).__init__(env, capacity)
        self.state_dims = []
        for obs in env.observation_space:
            self.state_dims.append(back_specified_dimension(obs))
        # 为了方便编码，暂时假定所有智能体的动作空间都是一样的！
        action = self.env.action_space[0]
        self.discrete = type(action) is Discrete
        if self.discrete:
            self.action_dim = action.n
            self.action_lim = -1
        else:
            self.action_dim = back_specified_dimension(env.action_space)
            self.action_lim = action.action_lim

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = 0.999
        self.tau = 0.001
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.agents = []
        self.use_global = use_global_state
        for i in range(self.env.agent_count):
            ag = DDPGAgent(self.state_dims[i], self.action_dim, self.action_lim,
                           self.learning_rate, self.discrete, capacity / self.env.agent_count,
                           self.device)
            self.agents.append(ag)
        return

    def get_exploitation_action(self, state):
        """
        得到给定状态下依据目标演员网络计算出的行为，不探索
        :param state: numpy数组
        :return: 动作 numpy数组
        """
        action_list = []
        for i in range(self.env.agent_count):
            s = flatten_data(state[i], self.state_dims[i], self.device)
            action = self.agents[i].step(s, explore=False)
            action_list.append(action)
        action_list = torch.vstack(action_list)
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
            action = self.agents[i].step(s, True if value < epsilon else False)
            action_list.append(action)
        action_list = torch.vstack(action_list)
        return action_list

    def _learn_from_memory(self):
        '''
        从记忆学习，更新两个网络的参数
        :return:
        '''
        # 随机获取记忆里的Transmition
        total_loss_actor = 0.0
        total_loss_critic = 0.0

        for i in range(self.env.agent_count):
            trans_pieces = self.agents[i].experience.sample(self.batch_size)

            s0 = np.array([x.s0 for x in trans_pieces])
            a0 = torch.vstack([x.a0 for x in trans_pieces]).detach()
            r1 = np.array([x.reward for x in trans_pieces])
            # is_done = np.array([x.is_done for x in trans_pieces])
            s1 = np.array([x.s1 for x in trans_pieces])
            global_state = np.array([x.global_state for x in trans_pieces])

            s0 = flatten_data(s0, self.state_dims[i], self.device, True)
            s1 = flatten_data(s1, self.state_dims[i], self.device, True)
            global_state = flatten_data(global_state, self.state_dims[-1], self.device, True)
            a1 = self.agents[i].target_actor.forward(s1).detach()
            r1 = torch.tensor(r1).type(torch.FloatTensor).to(self.device)

            # detach()的作用是让梯度无法传导到target_critic,因为此时只有critic需要更新！
            next_val = torch.squeeze(
                self.agents[i].target_critic.forward(s1
                                                     , a1)).detach()
            # 优化评判家网络参数，优化的目标是评判值与r+gamma*Q'(s1,a)尽量接近
            y_expected = r1 + self.gamma * next_val
            y_expected = y_expected.to(self.device)
            y_predicted = torch.squeeze(self.agents[i].critic.forward(s0, a0))  # 此时没有使用detach！
            loss_critic = F.smooth_l1_loss(y_predicted, y_expected).to(self.device)
            self.agents[i].critic_optimizer.zero_grad()
            loss_critic.backward()
            self.agents[i].critic_optimizer.step()
            total_loss_critic += loss_critic.item()

            # 优化演员网络参数，优化的目标是使得Q增大
            pred_a = self.agents[i].actor.forward(s0)
            # 反向梯度下降
            loss_actor = -1 * torch.sum(self.agents[i].critic.forward(s0, pred_a))

            self.agents[i].actor_optimizer.zero_grad()
            loss_actor.backward()
            self.agents[i].actor_optimizer.step()
            total_loss_actor += loss_actor.item()

            # 软更新参数
            soft_update(self.agents[i].target_actor, self.agents[i].actor, self.tau)
            soft_update(self.agents[i].target_critic, self.agents[i].critic, self.tau)
        return (total_loss_critic, total_loss_actor)

    def act(self, a0) -> tuple:
        '''
        给出行动a0，执行相应动作并返回参数
        :param a0:
        :return: s1(下一状态),r1(执行该动作环境给出的奖励),is_done(是否为终止状态),info,total_reward(目前episode的总奖励)
        '''
        s0 = self.state
        s1, r1, is_done, info = self.env.step(a0)
        total_rewards = []
        for i in range(self.env.agent_count):
            global_state = s1[-1] if self.use_global else None
            trans = Transition(s0[i], a0[i], r1[i], is_done[i], s1[i], global_state)
            r = self.agents[i].experience.push(trans)
            total_rewards.append(r)
        self.state = s1
        return s1, r1, is_done, info, total_rewards

    def learning_method(self, lambda_=0.9, gamma=0.9, alpha=0.5,
                        epsilon=0.2, explore=True, display=False,
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
            if self.total_trans > self.batch_size:
                loss_c, loss_a = self._learn_from_memory()
                loss_critic += loss_c
                loss_actor += loss_a
            self.state = s0
            time_in_episode += 1
            if wait:
                time.sleep(waitSecond)

        loss_critic /= time_in_episode
        loss_actor /= time_in_episode

        print("{}".format(self.last_episode_detail()))
        return time_in_episode, total_reward

    def play_init(self, savePath, s0):
        import os
        for i in range(self.env.agent_count):
            saPath = os.path.join(savePath, "Actor{}.pkl".format(i))
            self.load(saPath, self.agents[i].actor)
            hard_update(self.agents[i].target_actor, self.agents[i].actor)
        return self.get_exploitation_action(s0)

    def play_step(self, savePath, s0):
        return self.get_exploitation_action(s0)

    @property
    def total_trans(self) -> int:
        return sum([x.experience.total_trans for x in self.agents])

    def last_episode_detail(self):
        print("**************")
        for idx, ag in enumerate(self.agents):
            print("Agent {}:{}".format(idx,
                                       ag.experience.last_episode.__str__()))
