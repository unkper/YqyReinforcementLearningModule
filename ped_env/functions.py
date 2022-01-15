import random

import numpy as np

from math import cos, sin

ACTION_DIM = 8

def transfer_to_render(x,y,X,Y,scale=10.0):
    '''
    该函数将物理坐标转化为渲染坐标来进行输出
    :param x: 物体中心点坐标x
    :param y: 物体中心点坐标y
    :param X: 物体的宽度X
    :param Y: 物体的高度Y
    :param scale: 物理坐标与像素坐标的放大比例
    :return:
    '''
    x_, y_ = x - X / 2, y - Y / 2
    return x_ * scale, y_ * scale, X * scale, Y * scale

# 修复bug:未按照弧度值进行旋转
identity = np.array([1.0,0.0])
actions = [np.array([0.0,0.0])]
for angle in range(0, 360, int(360/ACTION_DIM)):#逆时针旋转angle的角度，初始为x轴向左
    theta = np.radians(angle)
    mat = np.array([[cos(theta), -sin(theta)],
                   [sin(theta), cos(theta)]])
    vec = np.squeeze((mat.dot(identity)).tolist())
    actions.append(np.array(vec))

def calculate_nij(i, j):
    pos_i = i.pos
    pos_j = j.pos
    return normalized(pos_i - pos_j)

from math import sqrt, acos
def angle_of_vector(v1, v2):
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180

def parse_discrete_action(type:np.ndarray):
    global actions
    return actions[np.argmax(type).item()]
    # sum_probabilities = sum(type)
    # for i in range(len(type)):
    #     type[i] /= sum_probabilities
    #return random_pick(actions, type)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return np.squeeze(a / np.expand_dims(l2, axis))

def calculate_each_group_num(group_size, person_num):
    '''
    根据人数随机计算每组人数的多少
    :param group_size:一个元组，代表一个组的范围
    :param person_num:需要分配的人总数
    :return:
    '''
    group_avg = int(sum(group_size) / 2)
    leader_num = int(person_num / group_avg)
    if leader_num == 0:
       raise Exception("Person_num must be bigger than group_avg_num!")
    else:
        group_num = []
        left_num = person_num
        for i in range(0, leader_num):
            num_in_one_group = random.randint(group_size[0], group_size[1])
            group_num.append(num_in_one_group)
            left_num -= num_in_one_group
        group_num[-1] += left_num
    return group_num

def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
         cumulative_probability += item_probability
         if x < cumulative_probability:
               break
    return item

def ij_power(r, A = 0.01610612736, B = 3.93216):
    ij_group_f =  (A / (pow(r, 12)) - B / (pow(r, 6)))
    return ij_group_f

if __name__ == '__main__':
    # for i in range(10):
    #     group_size = (5, 5)
    #     person_num = 10
    #     ret = calculate_each_group_num(group_size, person_num)
    #     print(ret)
    # Af, Bf = 0.4, 240
    # A = 4 * pow(Af, 12) * Bf
    # B = 4 * pow(Af, 6_map11_use) * Bf
    # sigma = pow(A / B, 1/6_map11_use)
    # mu = pow(B, 2)/(4*A)
    # print("Af={},Bf={},A={},B={},mu={},sigma={}".format(Af, Bf, A, B, mu, sigma))

    delta, counter = 0.01, 0
    start = 0.37
    x, y = [], []
    while counter <= 150:
        counter += 1
        r = (delta * counter) + start
        force = ij_power(r)
        x.append(counter / 100 + start)
        y.append(force)
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()