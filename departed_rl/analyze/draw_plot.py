import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from numpy import flipud, fliplr
from collections import defaultdict
from ped_env.utils.maps import *


colors = ['r', 'g', 'b', 'm', 'c']

def smooth_func(y, smooth_step):
    k = smooth_step
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return ysmoo

def set_plt_latex_form():
    #plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    #plt.rcParams['figure.dpi'] = 300  # 分辨率

def set_spine_visible(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

info_part = ['collision_agent_agent.npy', 'collision_wall_agent.npy', 'evacuation_time.npy', 'leader_pos.npy']

def load_info_data(dir, part, clip=-1):
    y = defaultdict(list)
    for folder in os.listdir(dir):
        pa = os.path.join(dir, folder)
        for label, folder_j in enumerate(os.listdir(pa)):
            pa1 = os.path.join(pa, folder_j, "extra_data")
            file = os.path.join(pa1, part)
            part_y = np.load(file)
            if clip != -1:
                part_y = np.where(part_y > 0, part_y, clip)
            y[label].append(part_y)
    data = []
    for label, da in y.items():
        da = np.mean(np.array(da), axis=0)
        data.append(da)
    return np.array(data)

def draw_explore_step_line(dir, calc=(0, 200), smooth_step=1, labels=None, ids=None):
    if labels is None:
        labels = ["GD-MAMBPO", "MAMBPO", "MATD3"]
    if ids is None:
        ids = list(range(len(labels)))
    global colors
    set_plt_latex_form()
    data = load_info_data(dir, info_part[2], clip=250)
    data = np.mean(data, axis=2)
    if calc != -1:
        data = data[:, calc[0]:calc[1]]
    plt.plot(figsize=(7.374, 4.626))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    fontsize = 25

    for i, id in zip(range(len(data)), ids):
        y = smooth_func(data[id], smooth_step)
        x = list(range(len(y)))  # 得到迭代的总步数
        plt.plot(x, y, color=colors[i], label=labels[i], linewidth=1.5)
    plt.xticks(fontsize=fontsize, )  # 默认字体大小为10
    plt.yticks(fontsize=fontsize, )

    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=fontsize, )  # 设置图例字体的大小和粗细 fontweight='bold'
    plt.xlabel("Episode", fontsize=fontsize, )
    plt.ylabel("Average Explore Step", fontsize=fontsize, )
    plt.xlim(5, )  # 设置x轴的范围
    plt.tight_layout()
    plt.show()
    #plt.ylim(1.5, 16)

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def draw_heatmap(dir, calc=(0, 200), agent_idx=None, clip = -1, single=False, labels=None, ids=None):
    set_plt_latex_form()
    if labels is None:
        labels = ["GD-MAMBPO", "MATD3"]
    if ids is None:
        ids = range(len(labels))
    data = load_info_data(dir, info_part[3])
    #取特定的人员索引
    agent_count = data.shape[2]
    if agent_idx == None:
        agent_idx = np.arange(agent_count)

    data = data[:, calc[0]:calc[1], agent_idx, :, :]
    data = np.squeeze(np.sum(data, 2, keepdims=True), 2) #将不同智能体所在位置加和
    data = np.squeeze(np.sum(data, 1, keepdims=True), 1) #将不同episode的数据相加
    data[:, 0, 0] = 0 #去掉默认地点为0的
    if clip != -1:
        data = np.where(data < clip, data, clip)

    print(np.sum(data, axis=(1,2)))
    ax_count = len(labels)
    for i in range(ax_count):
        data[i] /= np.sum(data[i])
    data = np.log(data)
    cmap = plt.cm.Reds
    if single:
        for i in range(ax_count):
            fig = plt.figure(figsize=(5,5))
            pcm = plt.pcolormesh(flipud(data[ids[i]].T), cmap = cmap)
            plt.title(labels[i])
            #plt.tight_layout()
            plt.show()
    else:
        fig, ax_arr = plt.subplots(1, ax_count, figsize=(23.42/2.52, 12.7/2.52))
        for i in range(ax_count):
            ax = ax_arr[i]
            #ax.imshow(map.map, cmap=cmap)
            pcm = ax.pcolormesh(data[ids[i]].T, cmap = cmap)
            ax.set_title(labels[i], fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
            set_spine_visible(ax)
        fig.tight_layout()
        plt.show()

def load_numpy_data(file):
    arr = np.load(file)
    print(arr)

def draw_mt_learning_curve(dir, y_range = None, y_func=np.mean,
                           smooth_step=-1, single=True, ax=None,
                           idx = None, envName="PedsMoveEnv",
                           labels=None, ids=None, use_step=False):
    if labels is None:
        labels = ['GD_MAMBPO', 'MAMBPO', 'MATD3']
    if ids is None:
        ids = ['GD_MAMBPO', 'MAMBPO', 'MATD3']
    global colors
    ys = [[] for _ in range(len(ids))]
    #加载reward数据从各个文件夹中
    for folder in os.listdir(dir):
        pa = os.path.join(dir,folder)
        for i, la, id, file in zip(range(len(labels)), labels, ids, os.listdir(pa)):
            reward_txt = os.path.join(pa, file, "rewards_{}.txt".format(id))
            part_y = np.loadtxt(reward_txt, delimiter=",")
            ys[i].append(part_y)

    controller = plt if single else ax

    #对各个Agent的数据做变换(取平均,最大值...),然后做平滑操作
    for y, c, la in zip(ys, colors, labels):
        y = np.stack(y)
        if y_range != None:
            y = y[:,y_range[0]:y_range[1],:]
        out_data = []
        for i in range(y.shape[0]):
            if y_func != None:
                temp_y = np.zeros(y.shape[1])
                for j in range(y.shape[1]):
                    temp_y[j] = y_func(y[i, j, :])
                if smooth_step != -1:
                    temp_y = smooth_func(temp_y, smooth_step)
                out_data.append(temp_y.tolist())
        x = np.arange(0, len(out_data[0]))
        if single:
            sns.tsplot(time=x, data=out_data, color=c,)# linestyle='-'
        else:
            sns.tsplot(time=x, data=out_data, color=c, ax=ax, linewidth=0.8)
    controller.locator_params(tight=True, nbins = 4)
    desc_font_size = 20
    axis_font_size = 18
    tick_font_size = 18
    if not single and idx == 0:
        controller.set_ylabel("Avarage Reward", fontdict={'fontsize':axis_font_size})
    if not single:
        controller.set_xlabel("Episode", fontdict={'fontsize':axis_font_size})
        controller.set_title(envName, fontdict={'fontsize':desc_font_size})
        controller.tick_params(axis="x", labelsize=tick_font_size)
        controller.tick_params(axis="y", labelsize=tick_font_size)
    controller.grid(True, linestyle='-.')
    if single:
        controller.legend(labels)
        controller.show()

def draw_long_pic_for_paper(dirs, y_range, smooth_step):
    labels = ['GD-MAMBPO', 'MAMBPO', 'MATD3']
    set_plt_latex_form()
    fig, ax_arr = plt.subplots(1, 3, constrained_layout=False, figsize=(17.05, 5.52))#figsize=(17.05, 5.52)
    envNames = ['4e','2b4e','6b4e']
    for i, name, dir in zip(range(len(envNames)), envNames, dirs):
        draw_mt_learning_curve(dir, y_range=y_range, smooth_step=smooth_step, labels=labels,
                               y_func=np.mean, single=False, ax=ax_arr[i], idx=i, envName=name)
    plt.tight_layout()
    plt.subplots_adjust(left=0.045, right=0.991, bottom=0.173, top=0.936, hspace=0.2, wspace=0.11)
    fig.legend(labels, loc="lower center", ncol=3, fontsize=20)

    plt.show()

def run2():
    #draw_heatmap("data/20220104/6_map11_use/train_11_no_avarage", calc=(0, 200), agent_idx=[1], clip = -1, single=False)
    #draw_explore_step_line("data/20220104/6_map11_use/train_11", calc=(0, 150), smooth_step=10)
    #load_info_data("./data/20220102/2", info_part[2])
    #load_numpy_data("./data/20220103/extra_data/leader_pos.npy")

    #draw_explore_step_line("data/20220108/bc_train_1", calc=(0, 80), smooth_step=10, labels=["BC-TD3", "TD3"])
    #draw_heatmap("data/20220108/bc_train_1", calc=(0, 80), agent_idx=[3,], labels=["BC-TD3", "TD3"])

    #draw_explore_step_line("data/20220104/6_map11_use/train_11", calc=(0, 400), smooth_step=10)
    draw_heatmap("data/20220104/6_map11_use/train_11", calc=(0, 400),
                 agent_idx=[7,], labels=["GD-MAMBPO","MATD3"], ids=[0,2])
    # draw_heatmap("data/20220104/6_map11_use/train_11_no_avarage", calc=(0, 400),
    #              agent_idx=[0, ], labels=["GD-MAMBPO", "MATD3"], ids=[0, 1])

    #draw_explore_step_line("data/20220108/bc_train_map_10", calc=(0, 80),
    #                       smooth_step=10, labels=["GD-MAMBPO","MAMBPO","MATD3-10","MATD3-1"])
    # draw_heatmap("data/20220108/bc_train_map_10", calc=(0, 80),
    #              agent_idx=[7,], labels=["GD_MAMBPO","MATD3"], ids=[0,2])
    #draw_explore_step_line("data/20220108/submit_curve_data/train_11", calc=(0, 400),
    #                   smooth_step=10, labels=["GD-MAMBPO","MAMBPO","MATD3-10"])
    # draw_heatmap("data/20220108/submit_curve_data/train_11", calc=(0, 400),
    #                     agent_idx=[3,], labels=["GD-MAMBPO","MAMBPO","MATD3-10"])


if __name__ == '__main__':
    # dirs = ["data/20211230_use/train_10", "data/20211230_use/train_11", "data/20211230_use/train_12"]
    #dirs = ["data/20220108/submit_curve_data/train_10", "data/20220108/submit_curve_data/train_11", "data/20220108/submit_curve_data/train_12"]
    #draw_long_pic_for_paper(dirs, y_range=(0, 400), smooth_step=10)
    # draw_mt_learning_curve("./data/20220108/submit_curve_data/train_12", y_range=(0, 400),
    #                        smooth_step=20, y_func=np.mean, labels=["MAMBPO","MATD3"], ids=["MAMBPO","MATD3"])
    # draw_mt_learning_curve("./data/20220108/bc_train_map_10", y_range=(0, 80), smooth_step=10,
    #                        y_func=np.mean, labels=["GD-MAMBPO","MAMBPO","MATD3-10","MATD3-1"], ids=["GD_MAMBPO","MAMBPO","MATD3","MATD3"])
    # draw_mt_learning_curve("./data/20220108/submit_curve_data/train_12", y_range=(0, 400), smooth_step=10,
    #                        y_func=np.mean, labels=["MAMBPO","MATD3-10"], ids=["MAMBPO","MATD3"])
    run2()
