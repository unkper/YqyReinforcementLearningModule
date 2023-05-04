import os
import pprint
import dill
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def draw_func(func):
    # 创建数据点
    x = np.linspace(1, 1000, 100)
    y = func(x)

    # 绘制图像
    plt.plot(x, y)

    # 添加标题和轴标签
    plt.title('1/N^{}'.format(phi))
    plt.xlabel('N')
    plt.ylabel('1/N^{}'.format(phi))

    # 显示图像
    plt.show()




def analyse(model_path):
    # 字典结构是先agent_id,然后
    with open(model_path, "rb") as model_file:
        model_dict = dill.load(model_file)
        agent_list = []
        for agentid in range(len(model_dict)):
            point_dict = defaultdict(int)
            for envid in range(len(model_dict[agentid])):
                for ele in model_dict[agentid][envid]:
                    point_dict[tuple(ele)] += 1
            agent_list.append(point_dict)
        v = list(agent_list[0].values())
        print(np.mean(v))
        print(np.max(v))
        print(np.median(v))

def delete_all_incr(pth):
    for folder in next(os.walk(pth))[1]:
        n_pth = os.path.join(pth, folder, "incremental")
        



if __name__ == '__main__':
    pth = r"D:\projects\python\PedestrainSimulationModule\third_party\maicm\models\pedsmove\map_09_4agents_taskleave\2023_04_29_00_15_47exp_test"
    #delete_all_incr(pth)

    # import dill

    phi = 0.7
    N = 10
    # # draw_func(lambda x: np.power(x, -phi))
    #
    #draw_func(lambda x: N / (N + np.power(x, phi)))
    #
    phi = 0.1
    additional = 0.1
    delta = 100
    func2 = lambda x: additional + 1 - 1 / (1 + np.exp(-phi * (x - delta)))
    draw_func(func2)

# analyse(r"D:\projects\python\PedestrainSimulationModule\third_party\maicm\models\pedsmove\map_09_4agents_taskleave\2023_04_29_00_15_47exp_test\run1\data\agent_pos.pkl")
