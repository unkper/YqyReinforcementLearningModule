import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import Tensor

from rl.utils.networks.pd_network import EPS

class MaddpgLstmCritic(nn.Module):
    def __init__(self, state_dims, action_dims):
        super(MaddpgLstmCritic, self).__init__()
        input_dim = sum(state_dims) + sum(action_dims)

        self.layer_sa2h = nn.Linear(input_dim, 64)
        self.layer_h2h2 = nn.LSTM(64, 64, batch_first=True)
        self.layer_h22o = nn.Linear(64, 1)

        self.no_linear = F.relu

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        h1 = self.no_linear(self.layer_sa2h(x))
        h2, _ = self.no_linear(self.layer_h2h2(h1))
        out = self.layer_h22o(h2)
        return out

class MaddpgLstmActor(nn.Module):
    def __init__(self, state_dim, action_dim, discrete):
        super(MaddpgLstmActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.LSTM(64, 64, batch_first=True)
        self.fc3 = nn.Linear(64, self.action_dim)
        self.no_linear = F.relu

        if self.discrete:
            print("离散动作,采用softmax作为输出!")
            self.out_fc = lambda x:x
        else:
            self.fc2.weight.data.uniform_(-EPS, EPS)
            self.out_fc = torch.tanh

    def forward(self,state) -> Tensor:
        '''
        前向运算，根据状态的特征表示得到具体的行为值
        :param state: 状态的特征表示 Tensor [n,state_dim]
        :return: 行为的特征表示 Tensor [n,action_dim]
        '''
        x = self.no_linear(self.fc1(state))
        x, _ = self.no_linear(self.fc2(x))
        action = self.out_fc(self.fc3(x))
        return action

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
    for i in tqdm.tqdm(range(100)):
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