import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

EMBEDDING_DIM = 512
TOP_HIDDEN = 4


class TopNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.bn_relu_for_dense = nn.Sequential(nn.Linear(EMBEDDING_DIM*2, EMBEDDING_DIM),
                                               nn.BatchNorm1d(EMBEDDING_DIM),
                                               nn.ReLU(inplace=True))
        self.linears = []
        for _ in range(TOP_HIDDEN):
            self.linears.append(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM))
            self.linears.append(nn.Sequential(nn.BatchNorm1d(EMBEDDING_DIM),
                                              nn.ReLU(inplace=True)))
        self.linears = nn.ModuleList(self.linears)
        self.output = nn.Linear(EMBEDDING_DIM, 2)
        self.softmax = nn.Softmax()

    def forward(self, x1, x2):
        x1 = torch.as_tensor(x1, device=self.device, dtype=torch.float32)
        x2 = torch.as_tensor(x2, device=self.device, dtype=torch.float32)
        bn = x1.data.shape[0]
        x = torch.concatenate([x1, x2], dim=1)
        x = self.bn_relu_for_dense(x)
        for module in self.linears:
            x = module(x)
        return self.softmax(self.output(x))


class Siamese_Resnet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.branch = models.resnet18()
        self.branch.fc = nn.Linear(512, EMBEDDING_DIM)
        self.similarity_network: TopNetwork = TopNetwork(device)

    def forward(self, x1, x2):
        x1 = torch.as_tensor(x1, device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
        x2 = torch.as_tensor(x2, device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
        y1 = self.branch(x1)
        y2 = self.branch(x2)
        return self.similarity_network(y1, y2)


from rl_platform.tianshou_case.third_party.pytorch_fitmodule.fit_module import FitModule


class RNetwork(FitModule):
    def __init__(self, input_shape, device):
        super().__init__()
        self.device = device
        self.net: Siamese_Resnet = Siamese_Resnet(device)
        (self._r_network, self._embedding_network,
         self._similarity_network) = (self.net, self.net.branch, self.net.similarity_network)

    def embed_observation(self, x):
        if isinstance(x, np.ndarray) and len(x.shape) == 3:  # 添加batch维度
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32).permute(2, 0, 1)
            x = torch.unsqueeze(x, 0)
        else:
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
        return self._embedding_network.forward(x)

    def embedding_similarity(self, x, y):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        return self._similarity_network.forward(x, y)[:, 1]

    def forward(self, x1, x2):
        x1 = self.embed_observation(x1)
        x2 = self.embed_observation(x2)
        return self.embedding_similarity(x1, x2)
