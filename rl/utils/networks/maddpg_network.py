import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaddpgLstmCritic(nn.Module):
    def __init__(self, state_dims, action_dims):
        super(MaddpgLstmCritic, self).__init__()
        input_dim = sum(state_dims) + sum(action_dims)

        self.layer_sa2h = nn.Linear(input_dim, 64)
        self.layer_h2h2 = nn.LSTM()