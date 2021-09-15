import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from torch import Tensor
from rl.utils.inits import fanin_init,random_init

EPS = 0.003

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self,input):
        out=input.view(input.size(0),-1)
        return out

class NetApproximator(nn.Module):
    def __init__(self,input_dim = 1, output_dim = 1, hidden_dim = 32):
        super(NetApproximator, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim,hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim,output_dim)
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    def _prepare_data(self, x, require_grad = False)->Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, int):
            x = torch.Tensor([[x]])
        x.requires_grad_ = require_grad
        x = x.float()
        if x.data.dim() == 1:
            x = x.unsqueeze(0) #torch.nn接收的数据都是2维的
        return x

    def to_device(self, data):
        if type(data) in [int, float]:
            data = torch.tensor(data).to(self.device)
        elif type(data) is np.ndarray:
            data = torch.from_numpy(data).to(self.device)
        else:
            data = data.to(self.device)
        return data

    def to_cpu(self,data:torch.Tensor):
        return data.cpu()

    def forward(self,x) -> Tensor:
        '''
        前向计算，根据输入x得到输出
        :param x:
        :return:
        '''
        x = self._prepare_data(x)
        x = self.to_device(x)
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        y_pred = self.to_cpu(y_pred)
        return y_pred

    def compute_grad(self,x,y,epochs = 1,criterion = None):
        if criterion is None:
            criterion = torch.nn.MSELoss()
        for t in range(epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred, y)

    def fit(self,x,y,criterion = None,optimizer = None,
            epochs=1, learning_rate=1e-4):
        '''
        通过训练更新权值w来拟合给定的输入x和输出y
        :param x:
        :param y:
        :param criterion:
        :param optimizer:
        :param epochs:
        :param learning_rate:
        :return:
        '''
        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(),lr = learning_rate)
        if epochs < 1:epochs = 1
        y = self._prepare_data(y,require_grad=False)
        y = self.to_device(y)
        x = self.to_device(x)
        for t in range(epochs):
            y_pred = self.forward(x)#前向传播
            y_pred = self.to_device(y_pred)
            loss = self.criterion(y_pred,y) #计算损失
            self.optimizer.zero_grad() #梯度重置，准备接受新梯度值
            loss.backward() # 反向传播时自动计算相应节点的梯度
            self.optimizer.step()
        return loss

    def __call__(self, x):
        y_pred = self.forward(x)
        return y_pred.data.numpy()

    def clone(self):
        return copy.deepcopy(self)

class Net_Paper2019Nov(nn.Module):
    def __init__(self,input_dim = 1, output_dim = 1):
        super(Net_Paper2019Nov, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim,64)
        self.linear2 = torch.nn.Linear(64,128)
        self.linear3 = torch.nn.Linear(128,64)
        self.linear4 = torch.nn.Linear(64,output_dim)
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    def _prepare_data(self, x, require_grad = False)->Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, int):
            x = torch.Tensor([[x]])
        x.requires_grad_ = require_grad
        x = x.float()
        if x.data.dim() == 1:
            x = x.unsqueeze(0) #torch.nn接收的数据都是2维的
        return x

    def to_device(self, data):
        if type(data) in [int, float]:
            data = torch.tensor(data).to(self.device)
        elif type(data) is np.ndarray:
            data = torch.from_numpy(data).to(self.device)
        else:
            data = data.to(self.device)
        return data

    def to_cpu(self,data:torch.Tensor):
        return data.cpu()

    def forward(self,x) -> Tensor:
        '''
        前向计算，根据输入x得到输出
        :param x:
        :return:
        '''
        x = self._prepare_data(x)
        x = self.to_device(x)
        h_relu_1 = F.relu(self.linear1(x))
        h_relu_2 = F.relu(self.linear2(h_relu_1))
        h_relu_3 = F.relu(self.linear3(h_relu_2))
        y_pred = self.linear2(h_relu_3)
        y_pred = self.to_cpu(y_pred)
        return y_pred

    def compute_grad(self,x,y,epochs = 1,criterion = None):
        if criterion is None:
            criterion = torch.nn.MSELoss()
        for t in range(epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred, y)

    def fit(self,x,y,criterion = None,optimizer = None,
            epochs=1, learning_rate=1e-4):
        '''
        通过训练更新权值w来拟合给定的输入x和输出y
        :param x:
        :param y:
        :param criterion:
        :param optimizer:
        :param epochs:
        :param learning_rate:
        :return:
        '''
        if criterion is None:
            criterion = torch.nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(),lr = learning_rate)
        if epochs < 1:epochs = 1
        y = self._prepare_data(y,require_grad=False)
        y = self.to_device(y)
        x = self.to_device(x)
        for t in range(epochs):
            y_pred = self.forward(x)#前向传播
            loss = criterion(y_pred,y) #计算损失
            optimizer.zero_grad() #梯度重置，准备接受新梯度值
            loss.backward() # 反向传播时自动计算相应节点的梯度
            optimizer.step()
        return loss

    def __call__(self, x):
        y_pred = self.forward(x)
        return y_pred.data.numpy()

    def clone(self):
        return copy.deepcopy(self)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        '''
        构建一个评判家网络
        :param state_dim: 状态的特征的数量
        :param action_dim: 行为作为输入的特征的数量
        '''
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.fcs1 = nn.Linear(state_dim,256) #状态的第一次线性变换
        self.fcs1.weight.data = random_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256,128) #状态第二次线性变换
        self.fcs2.weight.data = random_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(action_dim,128) # 行为第一次线性变换
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)  # (􀀎 􀀏+􀀴 􀀕)􀂕 􀁶 􀀉 􀂑 􀂒 􀁊 􀂓 􀀒 􀂖 􀂗 􀁽 􀁾 􀂘
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 1)  # (􀀎 􀀏+􀀴 􀀕)􀂕 􀁶 􀀉 􀂑 􀂒 􀁊 􀂓
        self.fc3.weight.data.uniform_(-EPS,EPS)

    def forward(self,state,action:Tensor)-> Tensor:
        '''
        前向运算，根据状态和行为的特征得到评判家给出的价值
        :param state: 状态的特征表示 Tensor [n,state_dim]
        :param action: 行为的特征表示 Tensor [n,action_dim]
        :return: Q(s,a) Tensor [n,1]
        '''
        device = self.device
        state = state.type(torch.FloatTensor).to(device)

        action = action.type(torch.FloatTensor).to(device)
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))

        a1 = F.relu(self.fca1(action))
        #将状态与行为连接起来
        x = torch.cat((s2,a1),dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        '''
        构建一个演员模型
        :param state_dim: 状态的特征数量 (int)
        :param action_dim: 行为作为输入的特征数量 (int)
        :param action_lim: 行为值的限定范围 [-action_lim, action_lim]
        '''
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(self.state_dim,256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128,64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(64,self.action_dim)
        self.fc4.weight.data.uniform_(-EPS,EPS)

    def forward(self,state) -> Tensor:
        '''
        前向运算，根据状态的特征表示得到具体的行为值
        :param state: 状态的特征表示 Tensor [n,state_dim]
        :return: 行为的特征表示 Tensor [n,action_dim]
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x)) #输出范围-1,1
        action = action * self.action_lim # 更改输出范围
        return action

class SimpleCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        '''
        构建一个评判家网络
        :param state_dim: 状态的特征的数量
        :param action_dim: 行为作为输入的特征的数量
        '''
        super(SimpleCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim,64) #状态的第一次线性变换

        self.fca1 = nn.Linear(action_dim,32) # 行为第一次线性变换


        self.fc3 = nn.Linear(96, 1)  # (􀀎 􀀏+􀀴 􀀕)􀂕 􀁶 􀀉 􀂑 􀂒 􀁊 􀂓

    def forward(self,state,action:Tensor)-> Tensor:
        '''
        前向运算，根据状态和行为的特征得到评判家给出的价值
        :param state: 状态的特征表示 Tensor [n,state_dim]
        :param action: 行为的特征表示 Tensor [n,action_dim]
        :return: Q(s,a) Tensor [n,1]
        '''
        s1 = F.relu(self.fcs1(state))
        a1 = F.relu(self.fca1(action))
        #将状态与行为连接起来
        x = torch.cat((s1,a1),dim=1)
        x = self.fc3(x)
        return x

class MADDPG_Critic(nn.Module):
    def __init__(self,state_dims:list,action_dims:list):
        super(MADDPG_Critic, self).__init__()
        input_dim = sum(state_dims) + sum(action_dims)

        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.out_layer = nn.Linear(64, 1)
        self.no_linear = F.relu

    def forward(self, state, action):
        temp = torch.cat([state, action],dim=1)
        h1 = self.no_linear(self.layer1(temp))
        h2 = self.no_linear(self.layer2(h1))
        h3 = self.out_layer(h2)
        return h3

class SimpleActor(nn.Module):
    def __init__(self, state_dim, action_dim, discrete, hidden_dim = 64,norm_in = True):
        '''
        构建一个演员模型
        :param state_dim: 状态的特征数量 (int)
        :param action_dim: 行为作为输入的特征数量 (int)
        :param action_lim: 行为值的限定范围 [-action_lim, action_lim]
        '''
        super(SimpleActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)
        self.no_linear = F.relu
        if self.discrete:
            print("离散动作,采用softmax作为输出!")
            self.out_fc = lambda x:x
        else:
            self.fc2.weight.data.uniform_(-EPS, EPS)
            self.out_fc = torch.tanh

    def forward(self,state) -> Tensor:
        '''
        前向运算，根据状态的特征表示得到具体的行为值
        :param state: 状态的特征表示 Tensor [n,state_dim]
        :return: 行为的特征表示 Tensor [n,action_dim]
        '''
        x = self.no_linear(self.fc1(state))
        x = self.no_linear(self.fc2(x))
        action = self.out_fc(self.fc3(x))
        return action

class SimpleActor02(nn.Module):
    def __init__(self, state_dim, action_dim, discrete, hidden_dim = 64,norm_in = True):
        '''
        构建一个演员模型
        :param state_dim: 状态的特征数量 (int)
        :param action_dim: 行为作为输入的特征数量 (int)
        :param action_lim: 行为值的限定范围 [-action_lim, action_lim]
        '''
        super(SimpleActor02, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, self.action_dim)
        if self.discrete:
            print("离散动作,采用softmax作为输出!")
            self.out_fc = lambda x:x
        else:
            self.fc2.weight.data.uniform_(-EPS, EPS)
            self.out_fc = torch.tanh

    def forward(self,state) -> Tensor:
        '''
        前向运算，根据状态的特征表示得到具体的行为值
        :param state: 状态的特征表示 Tensor [n,state_dim]
        :return: 行为的特征表示 Tensor [n,action_dim]
        '''
        x = F.relu(self.fc1(state))
        action = self.out_fc(self.fc2(x))
        return action