import torch
import torch.nn.functional as F

Q1 = torch.zeros([256, 1])
Q2 = torch.ones([256, 1])
print(F.mse_loss(Q1[Q2<Q1], Q2[Q2<Q1]))