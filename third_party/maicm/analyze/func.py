import os

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
            smoothed_data.append(sum(data[i-window:i]) / window)
    return smoothed_data


def draw_arrive_plot(path_dir, label_type=0, window=30):
    labels = {
        0 : ["with_intrinsic_reward",
             "without_intrinsic_reward"],
        1 : ["Independent exploration",
             "Minimum exploration",
             "Covering exploration",
             "Burrowing exploration",
             "Leader-Follower exploration",
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

    def _inner(y_label="total_n_found_exit"):
        fig, ax = plt.subplots()

        for fr, label in zip(dataframes, labels[label_type]):
            # 绘制图表
            ax.plot(fr['timestep'], fr[y_label].rolling(window).mean(), label=label)
        ax.set_xlabel('time_step')
        ax.set_ylabel(y_label.replace('/', '_'))
        ax.legend(loc="best")

        plt.savefig(os.path.join(path_dir, "plot_{}.png".format(y_label.replace("/", "_"))))
    _inner("total_n_found_exit")
    _inner("episode_rewards/extrinsic/mean")
    _inner("episode_lengths/mean")


if __name__ == '__main__':
    pth = r"/home/lab/projects/YqyReinforcementLearningModule/ec/maicm/models/pedsmove/map_10_5agents_taskleave/one_icm_test"
    draw_arrive_plot(pth)