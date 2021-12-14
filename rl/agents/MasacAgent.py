import random
import time

import torch

from gym import Env
from gym.spaces import Discrete
from torch.optim import Adam

from rl.agents.Agent import Agent
from rl.utils.networks.maddpg_network import GaussianPolicy, DoubleQNetworkCritic
from rl.utils.updates import soft_update, hard_update
from rl.utils.classes import SaveNetworkMixin, Noise, Experience, MAAgentMixin
from rl.utils.functions import back_specified_dimension, onehot_from_logits, gumbel_softmax, flatten_data, \
    onehot_from_int, save_callback, process_maddpg_experience_data, loss_callback

MSELoss = torch.nn.MSELoss()

class DDPGAgent:
    def __init__(self, state_dim, action_dim,
                 learning_rate, discrete,
                 device, state_dims, action_dims,
                 actor_network = None, critic_network = None,
                 actor_hidden_dim=64, critic_hidden_dim=64,
                 automatic_entropy_tuning=True):
        if discrete: raise Exception("只能处理连续动作空间!")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        self.actor = GaussianPolicy(state_dim, action_dim, actor_hidden_dim, discrete).to(self.device) \
            if actor_network == None else actor_network(state_dim, action_dim, discrete, actor_hidden_dim).to(self.device)
        self.target_actor = GaussianPolicy(state_dim, action_dim, actor_hidden_dim, discrete).to(self.device) \
            if actor_network == None else actor_network(state_dim, action_dim, discrete, actor_hidden_dim).to(self.device)
        hard_update(self.target_actor, self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), learning_rate)

        self.critic = DoubleQNetworkCritic(state_dims, action_dims).to(self.device)\
            if critic_network == None else critic_network(state_dims, action_dims, critic_hidden_dim).to(self.device)
        self.target_critic = DoubleQNetworkCritic(state_dims, action_dims).to(self.device)\
            if critic_network == None else critic_network(state_dims, action_dims, critic_hidden_dim).to(self.device)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), learning_rate)

        if automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)

        self.noise = Noise(1 if self.discrete else action_dim)
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
            action, _, _ = self.actor.sample(obs)
            action = action.clamp(-1, 1)
        elif not explore and self.discrete:
            action = self.actor(torch.unsqueeze(obs, dim=0))  # 统一以一批次的形式进行输入
            action = onehot_from_logits(action)
            action = torch.squeeze(action).to(self.device)
        else:
            action = self.actor(torch.unsqueeze(obs, dim=0))
            action = action.clamp(-1, 1)
            action = torch.squeeze(action).to(self.device)
        self.count[torch.argmax(action).item()] += 1
        return action

class MASACAgent(MAAgentMixin, SaveNetworkMixin, Agent):
    loss_recoder = []

    def __init__(self, env: Env = None,
                 capacity=2e6,
                 n_rol_threads = 1,
                 batch_size=128,
                 learning_rate=1e-4,
                 update_frequent = 4,
                 debug_log_frequent = 500,
                 gamma = 0.95,
                 tau = 0.01,
                 K = 5,
                 actor_network = None,
                 critic_network = None,
                 hidden_dim = 64,
                 env_name = "training_env"
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
        super(MASACAgent, self).__init__(env, capacity, env_name=env_name,  gamma=gamma, n_rol_threads=n_rol_threads)
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
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.train_update_count = 0
        self.K = K
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.agents = []
        self.experience = Experience(capacity)
        self.alpha = torch.zeros(1, requires_grad=True, device=self.device).exp()
        for i in range(self.env.agent_count):
            ag = DDPGAgent(self.state_dims[i], self.action_dims[i],
                           self.learning_rate, self.discrete, self.device, self.state_dims,
                           self.action_dims,actor_network,critic_network)
            self.agents.append(ag)

        self.loss_callback_ = loss_callback
        self.save_callback_ = save_callback
        return

    def __str__(self):
        return "Masac"

    def _learn_from_memory(self):
        '''
        从记忆学习，更新两个网络的参数
        :return:
        '''
        # 随机获取记忆里的Transmition
        total_critic_loss = 0.0
        total_loss_actor = 0.0

        trans_pieces = self.experience.sample(self.batch_size)

        s0, a0, r1, is_done, s1, s0_critic_in, s1_critic_in = \
            process_maddpg_experience_data(trans_pieces, self.state_dims, self.env.agent_count, self.device)

        for i in range(self.env.agent_count):
            with torch.no_grad():
                if self.discrete:
                    a1 = torch.cat([onehot_from_logits(self.agents[j].target_actor.forward(s1[j])).to(self.device)
                                    for j in range(self.env.agent_count)],dim=1)
                else:
                    a1,
                    for j in range(self.env.agent_count):
                        self.agents[j].target_actor.forward(s1[j])
                    a1 = torch.cat([self.agents[j].target_actor.forward(s1[j])
                                    for j in range(self.env.agent_count)],dim=1)
                #计算熵值
                log_a1 = self.calculate_log_action_probs(a1)
                # detach()的作用是让梯度无法传导到target_critic,因为此时只有critic需要更新！
                target_Q1, target_Q2 = self.agents[i].target_critic.forward(s1_critic_in, a1)
            # 用两个目标价值网络来计算取其中最小者为TD目标
            target_V = (a1 * (torch.min(target_Q1, target_Q2) - self.alpha * log_a1)).sum(dim=1, keepdim=True)
            # 优化两个评判家网络参数，优化的目标是使评判值与r + gamma * Q'(s1,a1)尽量接近
            target_Q = r1[:,i] + self.gamma * target_V * torch.tensor(1 - is_done[:,i]).to(self.device)
            current_Q1, current_Q2 = self.agents[i].critic.forward(s0_critic_in, a0)  # 此时没有使用detach！
            critic_loss = MSELoss(current_Q1, target_Q) + MSELoss(current_Q2, target_Q)
            self.agents[i].critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].critic.parameters(), 0.5)
            self.agents[i].critic_optimizer.step()
            total_critic_loss += critic_loss.item()

            # 每隔K轮才对策略网络和目标网络进行一次更新
            if self.train_update_count % self.K == 0:
                # 优化演员网络参数，优化的目标是使得Q增大
                curr_pol_out = self.agents[i].actor.forward(s0[i])
                pred_a = []
                if self.discrete:
                    for j in range(self.env.agent_count):
                        pred_a.append(gumbel_softmax(curr_pol_out).to(self.device)
                                      if i == j else onehot_from_logits(self.agents[j].actor.forward(s0[j])).to(self.device))
                    pred_a = torch.cat(pred_a, dim=1)
                else:
                    pred_a = torch.cat([self.agents[j].actor.forward(s0[j])
                                    for j in range(self.env.agent_count)],dim=1)
                log_pred_a = self.calculate_log_action_probs(log_pred_a)
                # Expectations of entropies.
                entropies = -torch.sum(pred_a * log_pred_a, dim=1, keepdim=True)
                # 反向梯度下降
                Q = torch.sum(self.agents[i].critic.Q1(s0_critic_in, pred_a) * pred_a, dim=1, keepdim=True)
                actor_loss = -Q - self.alpha * entropies
                # actor_loss += (curr_pol_out**2).mean() * 1e-3

                self.agents[i].actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[i].actor.parameters(), 0.5)
                self.agents[i].actor_optimizer.step()
                total_loss_actor += actor_loss.item()

                # 软更新参数
                soft_update(self.agents[i].target_actor, self.agents[i].actor, self.tau)
                soft_update(self.agents[i].target_critic, self.agents[i].critic, self.tau)

                # if self.writer is not None:
                #     self.writer.add_scalars('agent/losses i' % i,
                #                        {'vf_loss': critic_loss,
                #                         'pol_loss': actor_loss},
                #                        self.total_steps_in_train)

        self.train_update_count += 1
        return (total_critic_loss, total_loss_actor)

