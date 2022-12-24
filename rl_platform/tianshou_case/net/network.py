from typing import List, Union, Optional, Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

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
