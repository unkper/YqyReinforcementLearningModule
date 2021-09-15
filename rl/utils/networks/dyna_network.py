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

        self.layer_h2s1 = nn.Linear(hidden_dim, state_dim)
        self.layer_h2r = nn.Linear(hidden_dim, 1)
        self.layer_h2i = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        h_s = self.layer_s2h(s)
        h_a = self.layer_a2h(a)
        h = F.tanh(h_s + h_a)

        reward = self.layer_h2r(h)
        is_done = self.layer_h2i(h)
        state_1 = self.layer_h2s1(h)

        return reward, is_done, state_1
