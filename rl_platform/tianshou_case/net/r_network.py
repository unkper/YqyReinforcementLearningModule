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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = torch.as_tensor(x1, device=self.device, dtype=torch.float32)
        x2 = torch.as_tensor(x2, device=self.device, dtype=torch.float32)
        x = torch.concat([x1, x2], dim=1)
        x = self.bn_relu_for_dense(x)
        for module in self.linears:
            x = module(x)
        return self.sigmoid(self.output(x)) # 二分类问题应该使用Sigmoid


class Siamese_Resnet(nn.Module):
    def __init__(self, channel, device):
        super().__init__()
        self.device = device
        self.branch = models.resnet18(pretrained=True)
        if channel != 3:
            self.branch.conv1 = torch.nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.branch.fc = nn.Linear(512, EMBEDDING_DIM)
        self.similarity_network: TopNetwork = TopNetwork(device)

    def forward(self, x1, x2):
        x1 = torch.as_tensor(x1, device=self.device, dtype=torch.float32)
        x2 = torch.as_tensor(x2, device=self.device, dtype=torch.float32)
        y1 = self.branch(x1)
        y2 = self.branch(x2)
        return self.similarity_network(y1, y2)


from rl_platform.tianshou_case.third_party.pytorch_fitmodule.fit_module import FitModule


class RNetwork(FitModule):
    def __init__(self, input_shape, device):
        super().__init__()
        h, w, c = input_shape
        self.device = device
        self.net: Siamese_Resnet = Siamese_Resnet(c, device)
        (self._r_network, self._embedding_network,
         self._similarity_network) = (self.net, self.net.branch, self.net.similarity_network)

    def embed_observation(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self._embedding_network.forward(x)

    def embedding_similarity(self, x, y):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        return self._similarity_network.forward(x, y)[:, 1]

    def forward(self, x1, x2):
        x1 = self.embed_observation(x1)
        x2 = self.embed_observation(x2)
        return self.embedding_similarity(x1, x2)
