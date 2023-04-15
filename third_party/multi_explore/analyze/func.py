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

    fig, ax = plt.subplots()

    for fr, label in zip(dataframes, labels[label_type]):
        # 绘制图表
        ax.plot(fr['total_n_found_exit'], label=label)
    ax.set_xlabel('time_step')
    ax.set_ylabel('total_n_found_exit')
    ax.legend(loc="best")

    plt.savefig("./plot.png")


if __name__ == '__main__':
    pth = r"/home/lab/projects/YqyReinforcementLearningModule/third_party/multi_explore/models/pedsmove/map_09_5agents_taskleave/one_icm_test"
    draw_arrive_plot(pth)