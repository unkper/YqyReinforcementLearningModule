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

identity = np.mat([1.0,0.0])
actions = [(0.0,0.0)]
for angle in range(0, 360, int(360/ACTION_DIM)):
    mat = np.mat([[cos(angle), -sin(angle)],
                   [sin(angle), cos(angle)]])
    vec = np.squeeze((identity * mat).tolist())
    actions.append(vec)

def parse_discrete_action(type:np.ndarray):
    #actions = [(0.0,0.0),(-1.0,0.0),(1.0,0.0),(0.0,1.0),(0.0,-1.0)]
    global actions
    return np.array(actions[np.argmax(type).item()])