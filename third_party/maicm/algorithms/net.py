import torch
from torch import nn


def standard(v):
    data_min = v.min()
    data_max = v.max()
    return (v - data_min) / (data_max - data_min) * 255


class Encoder(nn.Module):
    def __init__(self, state_shape, out_dim=128):
        super(Encoder, self).__init__()
        self.batch_norm = nn.BatchNorm2d(1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        l2_out_dim = self.layer2(self.layer1(torch.zeros((1,) + state_shape))).view(-1).shape[0]
        self.fc1 = nn.Linear(l2_out_dim, out_dim)

    def forward(self, x):
        out = self.layer1(self.batch_norm(torch.unsqueeze(x, 1)))
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
