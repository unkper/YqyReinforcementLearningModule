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
from rl.utils.networks.pd_network import MLPNetworkActor, MLPNetwork_MACritic
from rl.utils.updates import soft_update, hard_update
from rl.utils.classes import SaveNetworkMixin, OrnsteinUhlenbeckActionNoise, Experience, MAAgentMixin
from rl.utils.functions import back_specified_dimension, onehot_from_logits, gumbel_softmax, flatten_data, \
    onehot_from_int, save_callback, process_maddpg_experience_data

MSELoss = torch.nn.MSELoss()


class DDPGAgent:
    def __init__(self, state_dim, action_dim,
                 learning_rate, discrete,
                 device, state_dims, action_dims,
                 actor_network=None, critic_network=None, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        self.actor = MLPNetworkActor(state_dim, action_dim, discrete).to(self.device) \
            if actor_network is None else actor_network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_actor = MLPNetworkActor(state_dim, action_dim, discrete).to(self.device) \
            if actor_network is None else actor_network(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.target_actor, self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                learning_rate)
        self.critic = MLPNetwork_MACritic(state_dims, action_dims).to(self.device) \
            if critic_network is None else critic_network(state_dims, action_dims, hidden_dim).to(self.device)
        self.target_critic = MLPNetwork_MACritic(state_dims, action_dims).to(self.device) \
            if critic_network is None else critic_network(state_dims, action_dims, hidden_dim).to(self.device)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 learning_rate)
        self.noise = OrnsteinUhlenbeckActionNoise(1 if self.discrete else action_dim)
        self.count = [0 for _ in range(action_dim)]

    def step(self, obs, explore):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore : Whether to explore or not
            eps :
        Outputs:
            action (Pytorch Variable): Actions for this agent
        """
        if explore and self.discrete:
            action = onehot_from_int(random.randint(0, self.action_dim - 1), self.action_dim)  # 利用随机策略进行采样
        elif explore and not self.discrete:
            action = torch.Tensor(self.noise.sample()).to(self.device)
            action = action.clamp(-1, 1)
        elif not explore and self.discrete:
            action = self.actor(torch.unsqueeze(obs, dim=0))  # 统一以一批次的形式进行输入
            action = onehot_from_logits(action)
            action = torch.squeeze(action).to(self.device)
        else:
            action = self.actor(torch.unsqueeze(obs, dim=0))
            action = action.clamp(-1, 1)
        self.count[torch.argmax(action).item()] += 1
        return action


class MADDPGAgent(MAAgentMixin, SaveNetworkMixin, Agent):
    loss_recoder = []

    def __init__(self, env: Env = None,
                 capacity=2e6,
                 n_rol_threads=1,
                 batch_size=128,
                 learning_rate=1e-4,
                 update_frequent=50,
                 debug_log_frequent=500,
                 gamma=0.95,
                 tau=0.01,
                 actor_network=None,
                 critic_network=None,
                 hidden_dim=64,
                 env_name="training_env"
                 ):
        '''
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
        :param learning_rate:
        :param update_frequent:
        :param debug_log_frequent:
        :param gamma:
        '''
        if env is None:
            raise Exception("agent should have an environment!")
        super(MADDPGAgent, self).__init__(env, capacity, gamma=gamma)
        self.state_dims = []
        for obs in env.observation_space:
            self.state_dims.append(back_specified_dimension(obs))
        # 为了方便编码，暂时不允许出现动作空间有Box和Space的情况!
        action = self.env.action_space[0]
        self.discrete = type(action) is Discrete
        self.action_dims = []
        for action in env.action_space:
            if self.discrete:
                self.action_dims.append(action.n)
            else:
                self.action_dims.append(back_specified_dimension(action))
        self.n_rol_threads = n_rol_threads
        self.batch_size = batch_size
        self.update_frequent = update_frequent
        self.log_frequent = debug_log_frequent
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.agents = []
        self.experience = Experience(capacity)
        self.env_name = env_name
        self.hidden_dim = hidden_dim
        for i in range(self.env.agent_count):
            ag = DDPGAgent(self.state_dims[i], self.action_dims[i],
                           self.learning_rate, self.discrete, self.device, self.state_dims,
                           self.action_dims, actor_network, critic_network, hidden_dim)
            self.agents.append(ag)

        def loss_callback(agent: MADDPGAgent, loss):
            self.loss_recoder.append(list(loss))
            if agent.total_episodes_in_train % self.log_frequent == 0 \
                    and len(self.loss_recoder) > 0:
                arr = np.array(self.loss_recoder)
                print("Critic mean Loss:{},Actor mean Loss:{}"
                      .format(np.mean(arr[-self.log_frequent:-1, 0]), np.mean(arr[-self.log_frequent:-1, 1])))

        self.loss_callback_ = loss_callback
        self.save_callback_ = save_callback
        return

    def __str__(self):
        return "Maddpg"

    def _learn_from_memory(self):
        '''
        从记忆学习，更新两个网络的参数
        :return:
        '''
        # 随机获取记忆里的Transmition
        total_loss_critic = 0.0
        total_loss_actor = 0.0

        trans_pieces = self.experience.sample(self.batch_size)
        s0, a0, r1, is_done, s1, s0_temp_in, s1_temp_in, s0_critic_in, s1_critic_in = \
            process_maddpg_experience_data(trans_pieces, self.state_dims, self.env.agent_count, self.device)

        for i in range(self.env.agent_count):
            if self.discrete:
                a1 = torch.cat([onehot_from_logits(self.agents[j].target_actor.forward(s1_temp_in[j]).detach()).to(self.device)
                                for j in range(self.env.agent_count)],dim=1)
            else:
                a1 = torch.cat([self.agents[j].target_actor.forward(s1_temp_in[j]).detach()
                                for j in range(self.env.agent_count)],dim=1)
            r1 = torch.tensor(r1).float().to(self.device)
            # detach()的作用是让梯度无法传导到target_critic,因为此时只有critic需要更新！
            next_val = torch.squeeze(
                self.agents[i].target_critic.forward(s1_critic_in
                                                     , a1)).detach()
            # 优化评判家网络参数，优化的目标是使评判值与r + gamma * Q'(s1,a1)尽量接近
            y_expected = r1[:, i] + self.gamma * next_val * torch.tensor(1 - is_done[:, i]).to(self.device)
            y_predicted = torch.squeeze(self.agents[i].critic.forward(s0_critic_in, a0))  # 此时没有使用detach！
            loss_critic = MSELoss(y_predicted, y_expected).to(self.device)
            self.agents[i].critic_optimizer.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].critic.parameters(), 0.5)
            self.agents[i].critic_optimizer.step()
            total_loss_critic += loss_critic.item()

            # 优化演员网络参数，优化的目标是使得Q增大
            curr_pol_out = self.agents[i].actor.forward(s0_temp_in[i])
            pred_a = []
            if self.discrete:
                for j in range(self.env.agent_count):
                    pred_a.append(gumbel_softmax(curr_pol_out).to(self.device)
                                  if i == j else onehot_from_logits(self.agents[j].actor.forward(s0_temp_in[j])).to(
                        self.device))
                pred_a = torch.cat(pred_a, dim=1)
            else:
                pred_a = torch.cat([self.agents[j].actor.forward(s0_temp_in[j])
                                    for j in range(self.env.agent_count)], dim=1)
            # 反向梯度下降
            loss_actor = -1 * self.agents[i].critic.forward(s0_critic_in, pred_a).mean()
            loss_actor += (curr_pol_out ** 2).mean() * 1e-3

            self.agents[i].actor_optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].actor.parameters(), 0.5)
            self.agents[i].actor_optimizer.step()
            total_loss_actor += loss_actor.item()
        self.update_all_targets()
        return (total_loss_critic, total_loss_actor)

    def update_all_targets(self):
        for agent in self.agents:
            # 软更新参数
            soft_update(agent.target_actor, agent.actor, self.tau)
            soft_update(agent.target_critic, agent.critic, self.tau)

