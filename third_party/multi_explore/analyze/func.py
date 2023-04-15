import os

import matplotlib.pyplot as plt
import pandas as pd


def draw_arrive_plot(path_dir):
    # 获取所有子文件夹名称
    subfolders = next(os.walk(path_dir))[1]
    dataframes = []

    # 打印子文件夹名称
    for subfolder in subfolders:
        p = os.path.join(path_dir, subfolder, r"data\main.xlsx")
        try:
            frame = pd.read_excel(p)
            dataframes.append(frame)
        except FileNotFoundError:
            continue

    for fr in dataframes:
        # 绘制图表
        plt.plot(fr['total_n_found_exit'])
    plt.xlabel('time_step')
    plt.ylabel('total_n_found_exit')
    plt.show()


if __name__ == '__main__':
    draw_arrive_plot(
        r"D:\projects\python\PedestrainSimulationModule\third_party\multi_explore\models\pedsmove\map_09_6agents_taskleave\test")
