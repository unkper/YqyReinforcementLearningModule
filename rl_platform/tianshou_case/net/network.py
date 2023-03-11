from typing import List, Union, Optional, Any, Dict, Tuple, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn
from tianshou.data import Batch

from tianshou.utils.net.common import MLP


class MLPNetwork(nn.Module):
    def __init__(self, state_dim, output_dim, hidden_dim: List[int], device):
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_dim,
            device=device,
        ).to(device)
        self.output_dim = output_dim
        self.device = device

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if type(obs) is np.ndarray:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        else:
            obs = torch.as_tensor(obs.obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int],
            device: Union[str, int, torch.device] = "cpu",
            features_only: bool = False,
            output_dim: Optional[int] = None,
            layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True), nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, np.prod(action_shape)))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, output_dim)),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state

class MarioPolicyHead(nn.Module):
    def __init__(self,
                 channel: int,
                 height: int,
                 width: int,
                 device: Union[str, int, torch.device] = "cpu",
                 layer_init: Callable[[nn.Module], nn.Module] = lambda x: x
                 ):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True), nn.Flatten()
        )
        with torch.no_grad():
            self.mid_dim = np.prod(self.net(torch.zeros(1, channel, height, width)).shape[1:])
        self.lstm_layer = nn.LSTM(self.mid_dim, 256)
        self.output_dim = 256

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x1 = torch.unsqueeze(self.net(obs), 0)
        x2, ht = self.lstm_layer(x1)
        x2 = torch.squeeze(x2, 0)
        return x2, state


class VizdoomPolicyHead(nn.Module):
    def __init__(self,
                 channel: int,
                 height: int,
                 width: int,
                 device: Union[str, int, torch.device] = "cpu",
                 layer_init: Callable[[nn.Module], nn.Module] = lambda x: x):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True), nn.Flatten()
        )
        with torch.no_grad():
            self.mid_dim = np.prod(self.net(torch.zeros(1, channel, height, width)).shape[1:])
        self.output_dim = self.mid_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        #print(obs.shape)
        obs = torch.permute(torch.as_tensor(obs, device=self.device, dtype=torch.float32), (0, 3, 1, 2))
        x1 = self.net(obs)
        return x1, state


class MarioICMFeatureHead(nn.Module):
    def __init__(self,
                 channel: int,
                 height: int,
                 width: int,
                 device: Union[str, int, torch.device] = "cpu",
                 layer_init: Callable[[nn.Module], nn.Module] = lambda x: x
                 ):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True), nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, channel, height, width)).shape[1:])

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x1 = self.net(obs)
        return x1, state


class PedPolicyHead(nn.Module):
    def __init__(self,
                 channel,
                 height,
                 width,
                 device="cpu",
                 layer_init: Callable[[nn.Module], nn.Module] = lambda x: x
                 ):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True), nn.Flatten()
        )
        with torch.no_grad():
            self.mid_dim = np.prod(self.net(torch.zeros(1, channel, height, width)).shape[1:])
        self.lstm_layer = nn.LSTM(self.mid_dim, 256)
        self.output_dim = 256

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        self.lstm_layer.flatten_parameters()
        if isinstance(obs, Batch):
            obs = obs.obs
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x1 = torch.unsqueeze(self.net(obs), 0)
        x2, ht = self.lstm_layer(x1)
        x2 = torch.squeeze(x2, 0)
        return x2, state


class PedICMFeatureHead(nn.Module):
    def __init__(self,
                 channel,
                 height,
                 width,
                 device: Union[str, int, torch.device] = "cpu",
                 layer_init: Callable[[nn.Module], nn.Module] = lambda x: x
                 ):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU(inplace=True), nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, channel, height, width)).shape[1:])

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {}, ):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x1 = self.net(obs)
        return x1, state
