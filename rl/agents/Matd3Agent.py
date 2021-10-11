import random
import time

import torch
import torch.nn.functional as F
import numpy as np

from gym import Env
from gym.spaces import Discrete

from rl.agents.Agent import Agent
from rl.utils.networks.pd_network import SimpleActor, MADDPG_Critic
from rl.utils.updates import soft_update, hard_update
from rl.utils.classes import SaveNetworkMixin, OrnsteinUhlenbeckActionNoise, Experience
from rl.utils.functions import back_specified_dimension, onehot_from_logits, gumbel_softmax, flatten_data, onehot_from_int

MSELoss = torch.nn.MSELoss()

class DDPGAgent:
    def __init__(self, state_dim, action_dim,
                 learning_rate, discrete,
                 device, state_dims, action_dims,
                 actor_network = None, critic_network = None, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        self.actor = SimpleActor(state_dim, action_dim, discrete).to(self.device) \
            if actor_network == None else actor_network(state_dim, action_dim, discrete, hidden_dim).to(self.device)
        self.target_actor = SimpleActor(state_dim, action_dim, discrete).to(self.device) \
            if actor_network == None else actor_network(state_dim, action_dim, discrete, hidden_dim).to(self.device)
        hard_update(self.target_actor, self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                learning_rate)
        self.critic = MADDPG_Critic(state_dims,action_dims).to(self.device)\
            if critic_network == None else critic_network(state_dims, action_dims, hidden_dim).to(self.device)
        self.target_critic = MADDPG_Critic(state_dims,action_dims).to(self.device)\
            if critic_network == None else critic_network(state_dims, action_dims, hidden_dim).to(self.device)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 learning_rate)
        self.other_critic = MADDPG_Critic(state_dims, action_dims).to(self.device) \
            if critic_network == None else critic_network(state_dims, action_dims, hidden_dim).to(self.device)
        self.other_target_critic = MADDPG_Critic(state_dims, action_dims).to(self.device) \
            if critic_network == None else critic_network(state_dims, action_dims, hidden_dim).to(self.device)
        hard_update(self.other_target_critic, self.other_critic)
        self.other_critic_optimizer = torch.optim.Adam(self.other_critic.parameters(),
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
            action = self.actor(torch.unsqueeze(obs, dim=0)) #统一以一批次的形式进行输入
            action = onehot_from_logits(action)
            action = torch.squeeze(action).to(self.device)
        else:
            action = self.actor(torch.unsqueeze(obs, dim=0))
            action = action.clamp(-1, 1)
        self.count[torch.argmax(action).item()] += 1
        return action

class MATD3Agent(Agent, SaveNetworkMixin):
    loss_recoder = []

    def __init__(self, env: Env = None,
                 capacity=2e6,
                 batch_size=128,
                 learning_rate=0.001,
                 update_frequent = 50,
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
        super(MATD3Agent, self).__init__(env, capacity, gamma=gamma)
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
        self.batch_size = batch_size
        self.update_frequent = update_frequent
        self.log_frequent = debug_log_frequent
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.train_update_count = 0
        self.K = K
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.agents = []
        self.experience = Experience(capacity)
        self.env_name = env_name
        for i in range(self.env.agent_count):
            ag = DDPGAgent(self.state_dims[i], self.action_dims[i],
                           self.learning_rate, self.discrete, self.device, self.state_dims,
                           self.action_dims,actor_network,critic_network)
            self.agents.append(ag)

        def loss_callback(agent:MATD3Agent, loss):
            self.loss_recoder.append(list(loss))
            if agent.total_episodes_in_train % self.log_frequent == 0 \
                    and len(self.loss_recoder) > 0:
                arr = np.array(self.loss_recoder)
                print("Critic mean Loss:{},Actor mean Loss:{}"
                      .format(np.mean(arr[-self.log_frequent:-1, 0]),np.mean(arr[-self.log_frequent:-1, 1])))
        self.loss_callback_ = loss_callback
        def save_callback(agent:MATD3Agent, episode_num:int):
            if episode_num % 500 == 0:
                print("save network!......")
                for i in range(agent.env.agent_count):
                    agent.save(agent.init_time_str + "_" + agent.env_name ,"Actor{}".format(i), agent.agents[i].actor)
                    agent.save(agent.init_time_str + "_" + agent.env_name ,"Critic{}".format(i), agent.agents[i].critic)
        self.save_callback_ = save_callback
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

    def _learn_from_memory(self):
        '''
        从记忆学习，更新两个网络的参数
        :return:
        '''
        # 随机获取记忆里的Transmition
        total_loss_critic = 0.0
        total_loss_actor = 0.0

        trans_pieces = self.experience.sample(self.batch_size)

        s0 = np.array([x.s0 for x in trans_pieces])
        a0 = np.array([x.a0 for x in trans_pieces])
        r1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        s1 = np.array([x.s1 for x in trans_pieces])

        s0 = [np.stack(s0[:, j], axis=0) for j in range(self.env.agent_count)]
        s1 = [np.stack(s1[:, j], axis=0) for j in range(self.env.agent_count)]

        s0_temp_in = [flatten_data(s0[j], self.state_dims[j], self.device, ifBatch=True)
                      for j in range(self.env.agent_count)]

        s1_temp_in = [flatten_data(s1[j], self.state_dims[j], self.device, ifBatch=True)
                      for j in range(self.env.agent_count)]

        s0_critic_in = torch.cat([s0_temp_in[j] for j in range(self.env.agent_count)], dim=1)
        s1_critic_in = torch.cat([s1_temp_in[j] for j in range(self.env.agent_count)], dim=1)

        a0 = torch.from_numpy(
            np.stack([np.concatenate(a0[i, :]) for i in range(a0.shape[0])], axis=0).astype(float)) \
            .float().to(self.device)

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
                self.agents[i].target_critic.forward(s1_critic_in, a1)).detach()
            next_val_2 = torch.squeeze(
                self.agents[i].other_target_critic.forward(s1_critic_in, a1)).detach()
            # 用两个目标价值网络来计算取其中最小者为TD目标
            true_next_val = torch.min(next_val, next_val_2)
            # 优化两个评判家网络参数，优化的目标是使评判值与r + gamma * Q'(s1,a1)尽量接近
            y_expected = r1[:,i] + self.gamma * true_next_val * torch.tensor(1 - is_done[:,i]).to(self.device)
            y_predicted_1 = torch.squeeze(self.agents[i].critic.forward(s0_critic_in, a0))  # 此时没有使用detach！
            y_predicted_2 = torch.squeeze(self.agents[i].other_critic.forward(s0_critic_in, a0))
            loss_critic = MSELoss(y_predicted_1, y_expected) + MSELoss(y_predicted_2, y_expected)
            self.agents[i].critic_optimizer.zero_grad()
            self.agents[i].other_critic_optimizer.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].critic.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.agents[i].other_critic.parameters(), 0.5)
            self.agents[i].critic_optimizer.step()
            self.agents[i].other_critic_optimizer.step()
            total_loss_critic += loss_critic.item()
            # 每隔K轮才对演员网络进行一次更新
            if self.train_update_count % self.K == 0:
                # 优化演员网络参数，优化的目标是使得Q增大
                curr_pol_out = self.agents[i].actor.forward(s0_temp_in[i])
                pred_a = []
                if self.discrete:
                    for j in range(self.env.agent_count):
                        pred_a.append(gumbel_softmax(curr_pol_out).to(self.device)
                                      if i == j else onehot_from_logits(self.agents[j].actor.forward(s0_temp_in[j])).to(self.device))
                    pred_a = torch.cat(pred_a,dim=1)
                else:
                    pred_a = torch.cat([self.agents[j].actor.forward(s0_temp_in[j])
                                    for j in range(self.env.agent_count)],dim=1)
                # 反向梯度下降
                loss_actor = -1 * self.agents[i].critic.forward(s0_critic_in, pred_a).mean()
                loss_actor += (curr_pol_out**2).mean() * 1e-3

                self.agents[i].actor_optimizer.zero_grad()
                loss_actor.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[i].actor.parameters(), 0.5)
                self.agents[i].actor_optimizer.step()
                total_loss_actor += loss_actor.item()
        # 每隔K轮才对目标网络进行一次更新
        if self.train_update_count % self.K == 0:
            self.update_all_targets()
        self.train_update_count += 1
        return (total_loss_critic, total_loss_actor)

    def update_all_targets(self):
        for agent in self.agents:
            # 软更新参数
            soft_update(agent.target_actor, agent.actor, self.tau)
            soft_update(agent.target_critic, agent.critic, self.tau)
            soft_update(agent.other_target_critic, agent.other_critic, self.tau)

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

    def play_init(self, savePath, s0):
        import os
        for i in range(self.env.agent_count):
            saPath = os.path.join(savePath, "Actor{}.pkl".format(i))
            self.load(saPath, self.agents[i].actor)
            hard_update(self.agents[i].target_actor, self.agents[i].actor)
        return self.get_exploitation_action(s0)

    def play_step(self, savePath, s0):
        return self.get_exploitation_action(s0)
