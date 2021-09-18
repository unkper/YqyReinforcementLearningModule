import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class dyna_model_network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        '''
        reward, is_done都视为一维的
        :param state_dim:
        :param action_dim:
        '''
        super(dyna_model_network, self).__init__()
        self.layer_s2h = nn.Linear(state_dim, hidden_dim)
        self.layer_a2h = nn.Linear(action_dim, hidden_dim)
        #这里hidden_dim*2是因为要把s和a连接起来
        self.layer_h2s1 = nn.Linear(hidden_dim * 2, state_dim)
        self.layer_h2r = nn.Linear(hidden_dim * 2, 1)
        self.layer_h2i = nn.Linear(hidden_dim * 2, 1)

    def forward(self, s, a):
        a = torch.unsqueeze(a, dim=1)
        h_s = self.layer_s2h(s)
        h_a = self.layer_a2h(a)
        h = F.tanh(torch.cat([h_s, h_a],dim=1))

        reward = self.layer_h2r(h)
        is_done = self.layer_h2i(h)
        state_1 = self.layer_h2s1(h)

        return state_1, reward, is_done

class dyna_q_network(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim):
        super(dyna_q_network, self).__init__()
        self.layer_i2h1 = nn.Linear(input_dim, hidden_dim)
        self.layer_h12h2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_h22o = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h_1 = F.relu(self.layer_i2h1(x))
        h_2 = F.relu(self.layer_h12h2(h_1))
        h_o = self.layer_h22o(h_2)
        return h_o
