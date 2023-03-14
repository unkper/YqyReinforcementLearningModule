from typing import Union, Callable, Optional, Dict, Any

import numpy as np
import torch
from torch import nn


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.1)

class CarRacingPolicyHead(nn.Module):
    def __init__(self,
                 channel: int,
                 height: int,
                 width: int,
                 device: Union[str, int, torch.device] = "cpu",
                 layer_init: Callable[[nn.Module], nn.Module] = _weights_init
                 ):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(channel, 8, kernel_size=3, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(128, 256, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.mid_dim = np.prod(self.net(torch.zeros(1, channel, height, width)).shape[1:])
        self.output_dim = self.mid_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        obs = torch.permute(torch.as_tensor(obs, device=self.device, dtype=torch.float32), (0, 3, 1, 2))

        #print("policy:"+str(obs.shape))
        x1 = self.net(obs)
        return x1, state