from typing import Union, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18


class MarioFeatureNet(nn.Module):
    def __init__(self,
                 channel: int,
                 height: int,
                 width: int,
                 device: Union[str, int, torch.device] = "cpu",
                 ):
        super().__init__()
        self.device = device
        self.net = resnet18(pretrained=False, num_classes=256)
        if channel != 3:
            self.net.conv1 = torch.nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.net = nn.Sequential(
        #     nn.Conv2d(channel, 8, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 16, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 32, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten()
        # )
        with torch.no_grad():
            self.mid_dim = np.prod(self.net(torch.zeros(1, channel, height, width)).shape[1:])
        self.output_dim = self.mid_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # print("policy:"+str(obs.shape))
        x1 = self.net(obs)
        return x1, state

    def eval(self):
        self.net.eval()


class MarioICMHead(MarioFeatureNet):
    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # print("policy:"+str(obs.shape))
        x1 = self.net(obs)
        return x1
