import random


from .miscellaneous import str_key

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

