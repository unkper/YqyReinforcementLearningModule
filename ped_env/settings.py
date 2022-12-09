from math import cos, sin

import numpy as np

TICKS_PER_SEC = 50  # 世界每一个step向前进 1 / TICKS_PER_SEC
vel_iters, pos_iters = 6, 2
ACTION_DIM = 9  # 离散动作的维数
GROUP_SIZE = 0.5  # 一个团体中的人在半径为多大(m)的区域生成

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
print(actions)