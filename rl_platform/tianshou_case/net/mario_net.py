from typing import Union, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(128 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 10 * 10)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MarioFeatureNet(nn.Module):
    def __init__(self,
                 channel: int,
                 height: int,
                 width: int,
                 device: Union[str, int, torch.device] = "cpu",
                 ):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten()
        )
        with torch.no_grad():
            self.mid_dim = np.prod(self.net(torch.zeros(1, channel, height, width)).shape[1:]).item()
        self.linear = nn.Linear(self.mid_dim, 256)
        self.output_dim = 256

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # print("policy:"+str(obs.shape))
        x1 = self.linear(self.net(obs))
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
        x1 = self.linear(self.net(obs))
        return x1
