import numpy as np
import torch
from torch import nn


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.model.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer