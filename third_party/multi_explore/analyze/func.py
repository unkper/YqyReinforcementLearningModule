import os

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def draw_arrive_plot(path_dir, label_type=0):
    labels = {
        0 : ["with_intrinsic_reward",
             "without_intrinsic_reward"],
        1 : ["Independent exploration",
             "Minimum exploration",
             "Covering exploration",
             "Burrowing exploration",
             "Leader-Follower exploration",
             "multihead"]
    }
    # 获取所有子文件夹名称
    subfolders = next(os.walk(path_dir))[1]
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
            ax.plot(fr['timestep'], fr[y_label], label=label)
        ax.set_xlabel('time_step')
        ax.set_ylabel(y_label.replace('/', '_'))
        ax.legend(loc="best")

        plt.savefig(os.path.join(path_dir, "plot_{}.png".format(y_label.replace("/", "_"))))
    _inner("total_n_found_exit")
    _inner("episode_rewards/extrinsic/mean")
    _inner("episode_lengths/mean")


if __name__ == '__main__':
    pth = r"/home/lab/projects/YqyReinforcementLearningModule/third_party/multi_explore/models/pedsmove/map_10_5agents_taskleave/one_icm_test"
    draw_arrive_plot(pth)