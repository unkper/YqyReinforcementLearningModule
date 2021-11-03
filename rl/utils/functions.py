import random

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from torch.autograd import Variable
from gym.spaces import Discrete, Box

from rl.utils.miscellaneous import str_key


def set_dict(target_dict, value, *args):
    if target_dict is None:
        return
    target_dict[str_key(*args)] = value

def get_dict(target_dict, *args):
    #print("key: {}".format(str_key(*args)))
    if target_dict is None:
        return
    return target_dict.get(str_key(*args),0)

def uniform_random_pi(A, s = None, Q = None, a = None):
    '''均一随机策略下某行为的概率
    '''
    n = len(A)
    if n == 0:
        return 0.0
    return 1.0/n

def sample(A):
    '''从A中随机选一个
    '''
    return random.choice(A) # 随机选择A中的一个元素

def greedy_pi(A, s, Q, a):
    '''依据贪婪选择，计算在行为空间A中，状态s下，a行为被贪婪选中的几率
    考虑多个行为的价值相同的情况
    '''
    #print("in greedy_pi: s={},a={}".format(s,a))
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:# 统计后续状态的最大价值以及到达到达该状态的行为（可能不止一个）
        q = get_dict(Q, s, a_opt)
        #print("get q from dict Q:{}".format(q))
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            #print("in greedy_pi: {} == {}".format(q,max_q))
            a_max_q.append(a_opt)
    n = len(a_max_q)
    if n == 0: return 0.0
    return 1.0/n if a in a_max_q else 0.0

def epsilon_greedy_pi(A, s, Q, a, epsilon = 0.1):
    m = len(A)
    if m == 0: return 0.0
    greedy_p = greedy_pi(A, s, Q, a)
    #print("greedy prob:{}".format(greedy_p))
    if greedy_p == 0:
        return epsilon / m
    n = int(1.0/greedy_p)
    return (1 - epsilon) * greedy_p + epsilon/m

def back_specified_dimension(space)->int:
    if type(space) is Discrete:
        return 1
    elif type(space) is Box:
        ret = 1
        for x in space.shape:
            ret *= x
        return ret
    else:
        raise Exception("目前只能处理Discete与Box类型的空间!")

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

def onehot_from_int(x,action_dim:int):
    #小数是为了能够做梯度计算！
    return torch.tensor([0.0 if i != x else 1.0 for i in range(action_dim)])

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits.cpu(), temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

def flatten_data(data, dim, device, ifBatch=False):
    #将状态数组打平!
    if type(data) is list:data = np.array(data)
    data = torch.from_numpy(data).float().to(device)
    if ifBatch:
        batchSize = data.shape[0]
        return data.reshape(batchSize,dim)
    else:
        return data.reshape(dim)

def process_experience_data(trans_pieces, to_tensor = False, device = None):
    states_0 = np.vstack([x.s0 for x in trans_pieces])
    actions_0 = np.array([x.a0 for x in trans_pieces])
    reward_1 = np.array([x.reward for x in trans_pieces])
    is_done = np.array([x.is_done for x in trans_pieces])
    states_1 = np.vstack(x.s1 for x in trans_pieces)

    if to_tensor:
        states_0 = torch.from_numpy(states_0).float().to(device)
        states_1 = torch.from_numpy(states_1).float().to(device)
        actions_0 = torch.from_numpy(actions_0).float().to(device)
        reward_1 = torch.from_numpy(reward_1).float()
        is_done = torch.from_numpy(is_done)
    return states_0,actions_0,reward_1,is_done,states_1

def process_maddpg_experience_data(trans_pieces, state_dims, agent_count, device = None):
    s0 = np.array([x.s0 for x in trans_pieces])
    a0 = np.array([x.a0 for x in trans_pieces])
    r1 = np.array([x.reward for x in trans_pieces])
    is_done = np.array([x.is_done for x in trans_pieces])
    s1 = np.array([x.s1 for x in trans_pieces])

    s0 = [np.stack(s0[:, j], axis=0) for j in range(agent_count)]
    s1 = [np.stack(s1[:, j], axis=0) for j in range(agent_count)]

    s0_temp_in = [flatten_data(s0[j], state_dims[j], device, ifBatch=True)
                  for j in range(agent_count)]

    s1_temp_in = [flatten_data(s1[j], state_dims[j], device, ifBatch=True)
                  for j in range(agent_count)]

    s0_critic_in = torch.cat([s0_temp_in[j] for j in range(agent_count)], dim=1)
    s1_critic_in = torch.cat([s1_temp_in[j] for j in range(agent_count)], dim=1)

    a0 = torch.from_numpy(
        np.stack([np.concatenate(a0[j, :]) for j in range(a0.shape[0])], axis=0).astype(float)) \
        .float().to(device)
    return s0,a0,r1,is_done,s1,s0_temp_in,s1_temp_in,s0_critic_in,s1_critic_in

def print_train_string(experience, episodes=500):
    rewards = []
    last_episodes = experience.last_n_episode(episodes if episodes > experience.len else experience.len)
    if last_episodes is None:
        print("episodes is None!!!")
        return
    rewards.append(np.mean([x.total_reward for x in last_episodes]))
    print("average rewards in last {} episodes:{}".format(episodes, rewards))
    print("{}".format(experience.__str__()))

def save_callback(agent, episode_num: int):
    sname = agent.init_time_str + "_" + agent.env_name
    if episode_num % 500 == 0:
        print("save network!......")
        for i in range(agent.env.agent_count):
            agent.save(sname, "Actor{}".format(i), agent.agents[i].actor)
            agent.save(sname, "Critic{}".format(i), agent.agents[i].critic)