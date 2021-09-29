import math

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



if __name__ == '__main__':
    rnn = torch.nn.LSTM(10,20,batch_first=True)

    h0 = torch.randn(1, 1, 20)
    c0 = torch.randn(1, 1, 20)
    for i in range(100):
        input = torch.unsqueeze(torch.unsqueeze(torch.sin(torch.range(i, i + 0.9 * math.pi, math.pi / 10)), dim = 0), dim = 0)
        output, (hn, cn) = rnn(input, (h0, c0))
        h0, c0 = hn, cn
    x = torch.unsqueeze(torch.unsqueeze(torch.sin(torch.range(0, 0.9 * math.pi, math.pi/10)), dim = 0), dim = 0)
    y , _ = rnn(x, (h0, c0))
    import matplotlib.pyplot as plt
    plt.plot(range(1, 10), y, marker="*")
    plt.savefig("sin.jpg")
    plt.show()