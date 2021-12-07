import numpy as np
import torch
from torch import nn


def fanin_init(size:list, fanin = None):
    '''
    给出数组大小与参数，返回经过一定规律初始化的数组
    :param size:
    :param fanin:
    :return:
    '''
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    x = torch.Tensor(size).uniform_(-v,v) #从-v到v的均匀分布
    return x.type(torch.FloatTensor)

def random_init(size):
    x = torch.Tensor(size).random_(-1,1)
    return x.type(torch.FloatTensor)

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)