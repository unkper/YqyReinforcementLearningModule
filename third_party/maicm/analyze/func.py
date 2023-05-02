import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def smoothed_moving_average(data, window=10):
    """
    使用滑动平均函数对数据进行平滑处理
    :param data: 待平滑的数据
    :param window: 平滑窗口大小
    :return: 平滑后的数据
    """
    smoothed_data = []
    for i in range(len(data)):
        if i < window:
            smoothed_data.append(data[i])
        else:
            smoothed_data.append(sum(data[i - window:i]) / window)
    return smoothed_data


def random_noise(data: pd.DataFrame, column_name, range=(-1, 1)):
    # 生成10个随机浮点数作为浮动数
    fluctuations = np.random.uniform(range[0], range[1], size=data.size)

    # 将浮动数添加到指定列中
    data[column_name] = data[column_name] + fluctuations
    return data


def draw_arrive_plot(path_dir, label_type=0, window=1, mrange=(-1, 1)):
    labels = {
        0: ["with_intrinsic_reward",
            "without_intrinsic_reward"],
        1: ["Independent exploration",
            "Minimum exploration",
            "Covering exploration",
            "Burrowing exploration",
            "Leader-Follower exploration",
            "A* Prior-Knowledge exploration"
            "multihead"],
        2: range(100)
    }
    # 获取所有子文件夹名称,按照run1,run2的方式排序,无法处理runXX两位数的情况!
    subfolders = sorted(next(os.walk(path_dir))[1])
    dataframes = []

    # 打印子文件夹名称
    for subfolder in subfolders:
        p = os.path.join(path_dir, subfolder, r"data/main.xlsx")
        try:
            frame = pd.read_excel(p)
            dataframes.append(frame)
        except FileNotFoundError:
            continue

    def _inner(y_label="total_n_found_exit", save_pth=path_dir):
        fig, ax = plt.subplots()

        for fr, label in zip(dataframes, labels[label_type]):
            # 绘制图表
            if window != 1:
                ax.plot(fr['timestep'], fr[y_label].rolling(window).mean(), label=label)
            else:
                ax.plot(fr['timestep'], fr[y_label], label=label)
        ax.set_xlabel('time_step')
        ax.set_ylabel(y_label.replace('/', '_'))
        ax.legend(loc="best")

        plt.savefig(os.path.join(save_pth, "plot_{}.png".format(y_label.replace("/", "_"))))

    s_pth = path_dir
    _inner("total_n_found_exit", s_pth)
    _inner("episode_rewards/extrinsic/mean", s_pth)
    _inner("episode_lengths/mean", s_pth)

    # 开始绘制500，1000，2000步数的折线图
    for st in [500, 1000, 2000]:
        dataframes = []
        s_pth = os.path.join(path_dir , str(st))
        if not os.path.exists(s_pth):
            os.makedirs(s_pth)
        for subfolder in subfolders:
            pth = os.path.join(path_dir, subfolder, r"data/main_step{}.xlsx".format(st))
            try:
                frame = pd.read_excel(pth)
                dataframes.append(frame)
            except FileNotFoundError:
                continue
        _inner("n_found_exits/extrinsic/mean", s_pth)
        _inner("step_rewards/extrinsic/mean", s_pth)


if __name__ == '__main__':
    pth = r"D:\projects\python\PedestrainSimulationModule\third_party\maicm\models\pedsmove\map_09_4agents_taskleave\2023_05_02_00_48_57exp_test"
    draw_arrive_plot(pth, 2)
