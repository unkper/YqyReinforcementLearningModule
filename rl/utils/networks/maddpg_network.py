import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.utils.networks.pd_network import EPS

class MLPNetwork_MACritic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dim=64):
        super(MLPNetwork_MACritic, self).__init__()
        input_dim = sum(state_dims) + sum(action_dims)

        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.out_layer = nn.Linear(64, 1)
        self.no_linear = F.relu

    def forward(self, state, action):
        temp = torch.cat([state, action],dim=1)
        h1 = self.no_linear(self.layer1(temp))
        h2 = self.no_linear(self.layer2(h1))
        h3 = self.out_layer(h2)
        return h3

class MLPNetworkActor(nn.Module):
    def __init__(self, state_dim, action_dim, discrete, hidden_dim = 64, norm_in = True):
        '''
        构建一个演员模型
        :param state_dim: 状态的特征数量 (int)
        :param action_dim: 行为作为输入的特征数量 (int)
        :param action_lim: 行为值的限定范围 [-action_lim, action_lim]
        '''
        super(MLPNetworkActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)
        self.no_linear = F.relu
        if self.discrete:
            print("离散动作,采用softmax作为输出!")
            self.out_fc = lambda x:x
        else:
            self.fc2.weight.data.uniform_(-EPS, EPS)
            self.out_fc = torch.tanh

    def forward(self,state):
        '''
        前向运算，根据状态的特征表示得到具体的行为值
        :param state: 状态的特征表示 Tensor [n,state_dim]
        :return: 行为的特征表示 Tensor [n,action_dim]
        '''
        x = self.no_linear(self.fc1(state))
        x = self.no_linear(self.fc2(x))
        action = self.out_fc(self.fc3(x))
        return action

class MaddpgLstmCritic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dim=64):
        super(MaddpgLstmCritic, self).__init__()
        input_dim = sum(state_dims) + sum(action_dims)

        self.layer_sa2h = nn.Linear(input_dim, hidden_dim)
        self.layer_h2h2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.layer_h22o = nn.Linear(hidden_dim, 1)
        self.no_linear = F.relu

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        h1 = self.no_linear(self.layer_sa2h(x))
        h2 = torch.unsqueeze(h1, dim=1)
        h2, _ = self.layer_h2h2(h2)
        h2 = torch.squeeze(h2, dim=1)
        h2 = self.no_linear(h2)
        out = self.layer_h22o(h2)
        return out

class MaddpgLstmActor(nn.Module):
    def __init__(self, state_dim, action_dim, discrete, hidden_dim=64):
        super(MaddpgLstmActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)
        self.no_linear = F.relu

        if discrete:
            print("离散动作,采用softmax作为输出!")
            self.out_fc = lambda x:x
        else:
            self.fc2.weight.data.uniform_(-EPS, EPS)
            self.out_fc = torch.tanh

    def forward(self,state):
        '''
        前向运算，根据状态的特征表示得到具体的行为值
        :param state: 状态的特征表示 Tensor [n,state_dim]
        :return: 行为的特征表示 Tensor [n,action_dim]
        '''
        x = self.no_linear(self.fc1(state))
        x = torch.unsqueeze(x, dim=1)
        x, _ = self.fc2(x)
        x = torch.squeeze(x, dim=1)
        x = self.no_linear(x)
        action = self.out_fc(self.fc3(x))
        return action

class G_MaddpgLstmModelNetwork(nn.Module):
    def __init__(self, state_dims:List[int], action_dims:List[int], hidden_dim):
        super(G_MaddpgLstmModelNetwork, self).__init__()
        agent_count = len(state_dims)
        vfs_in_dim = sum(state_dims)
        vfa_in_dim = sum(action_dims)
        self.layer_s2h = nn.Linear(vfs_in_dim, hidden_dim)
        self.layer_a2h = nn.Linear(vfa_in_dim, hidden_dim)
        #将两个向量拼接在一起
        self.layer_h2h2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.layer_h22s1 = nn.Linear(hidden_dim, vfs_in_dim)
        self.layer_h22r = nn.Linear(hidden_dim, agent_count)
        self.layer_h22i = nn.Linear(hidden_dim, agent_count)

    def forward(self, s, a):
        h_s = self.layer_s2h(s)
        h_a = self.layer_a2h(a)
        h = F.relu(torch.cat([h_s, h_a], dim=1))
        h = torch.unsqueeze(h, dim=1)
        h2, _ = self.layer_h2h2(h)
        h2 = F.relu(torch.squeeze(h2, dim=1))

        state_1 = self.layer_h22s1(h2)
        reward = self.layer_h22r(h2)
        is_done = self.layer_h22i(h2)

        return state_1, reward, is_done

if __name__ == '__main__':
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.rnn = torch.nn.LSTM(10,20,batch_first=True)
            self.linear = torch.nn.Linear(20,10)

        def forward(self, x):
            out_h1, (memory_h, memory_c) = self.rnn(x)
            out_y = self.linear(nn.functional.relu(out_h1))
            return out_y

    network = Network()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    import tqdm
    for i in tqdm.tqdm(range(10000)):
        input = torch.unsqueeze(torch.unsqueeze(torch.sin(torch.from_numpy(np.linspace(i, i + math.pi, 10)).float()), dim = 0), dim = 0)
        output = network(input)
        real = torch.unsqueeze(torch.unsqueeze(torch.sin(torch.from_numpy(np.linspace(i + math.pi, i + 2 * math.pi, 10)).float()), dim = 0), dim = 0)

        loss = torch.nn.functional.mse_loss(output, real)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    start = 0
    x = torch.unsqueeze(torch.unsqueeze(torch.sin(torch.from_numpy(np.linspace(start, start + math.pi, 10)).float()), dim = 0), dim = 0)
    out = network(x)
    real = torch.unsqueeze(torch.unsqueeze(torch.sin(torch.from_numpy(np.linspace(start + math.pi, start + 2 * math.pi, 10)).float()), dim = 0), dim = 0)
    x = torch.squeeze(x)
    out = torch.squeeze(out)
    real = torch.squeeze(real)
    import matplotlib.pyplot as plt
    plt.plot(np.linspace(start, start + 2 * math.pi, 20), torch.cat([x, out]).detach().numpy(), marker="*")
    plt.plot(np.linspace(start, start + 2 * math.pi, 20), torch.cat([x, real]).detach().numpy(), marker="h")
    plt.savefig("sin.jpg")
    plt.show()