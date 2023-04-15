import logging
from math import cos, sin

import numpy as np

TICKS_PER_SEC = 50  # 世界每一个step向前进 1 / TICKS_PER_SEC
vel_iters, pos_iters = 6, 2
ACTION_DIM = 9  # 离散动作的维数
GROUP_SIZE = 0.5  # 一个团体中的人在半径为多大(m)的区域生成


VIEWPORT_W, VIEWPORT_H = int(500), int(500)
RENDER_SCALE = int(85)
RENDER_RATIO_CHANGED = False

# 动作常量区
# 修复bug:未按照弧度值进行旋转
identity = np.array([1.0, 0.0])
actions = [np.array([0.0, 0.0])]
for angle in range(0, 360, int(360 / (ACTION_DIM - 1))):  # 逆时针旋转angle的角度，初始为x轴向左
    theta = np.radians(angle)
    mat = np.array([[cos(theta), -sin(theta)],
                    [sin(theta), cos(theta)]], dtype=np.float)
    vec = np.squeeze((mat.dot(identity)).tolist())
    actions.append(np.array(vec))
# print(actions)

identity = np.array([1, 0])
DIRECTIONS = [np.zeros([2])]
for idx, angle in enumerate(range(0, 360, int(360 / 8))):
    theta = np.radians(angle)
    mat = np.array([cos(theta), -sin(theta),
                    sin(theta), cos(theta)]).reshape([2, 2])
    vec = np.matmul(mat, identity)
    DIRECTIONS.append(vec)

logging.warning(u"在使用渲染前一定要调用init_settings初始化")


def init_settings(map_width, map_height):
    global VIEWPORT_H, VIEWPORT_W, RENDER_SCALE, RENDER_RATIO_CHANGED
#    assert map_width == map_height, u"当前地图必须保证输入的是正方形!"
    RENDER_SCALE = 500 // map_width
    RENDER_RATIO_CHANGED = True
