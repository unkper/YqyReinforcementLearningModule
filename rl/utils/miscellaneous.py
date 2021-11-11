import uuid
import os
import datetime
import numpy as np

import matplotlib.pyplot as plt

def str_key(*args):
    '''将参数用"_"连接起来作为字典的键，需注意参数本身可能会是tuple或者list型，
    比如类似((a,b,c),d)的形式。
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            if arg is None:
                pass
            else:
                new_arg.append(str(arg))
    return "_".join(new_arg)

def learning_curve(data, x_index = 0, y1_index = 1, y1_func = None, step_index = None, title = "",
                   x_name = "", y_name = "",
                   y1_legend = "", saveName = "picture",
                   save=True, save_dir="./", show=False):
    '''根据统计数据绘制学习曲线，
    Args:
        statistics: 数据元组，每一个元素是一个列表，各列表长度一致 ([], [], [])
        x_index: x轴使用的数据list在元组tuple中索引值
        y_index: y轴使用的数据list在元组tuple中的索引值
        title:图标的标题
        x_name: x轴的名称
        y_name: y轴的名称
        y1_legend: y1图例
        y2_legend: y2图例
    Return:
        None 绘制曲线图
    '''
    fig, ax = plt.subplots()
    x = np.array(data[x_index])
    y1 = np.array(data[y1_index])
    if y1_func != None:
        temp_y = np.zeros(y1.shape[0])
        for i in range(y1.shape[0]):
            temp_y[i] = y1_func(y1[i, :])
        y1 = temp_y
    if step_index != None:
        step = np.array(data[step_index])
        print("输出经过步数平均后的值!")
        for i in range(len(y1)):
            y1[i] = y1[i] / (step[i] if i == 0 else step[i] - step[i-1])
    ax.plot(x.data, y1.data, label = y1_legend)
    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='black', labelsize='medium', width=1)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend("rewards")

    if save:
        plt.savefig(os.path.join(save_dir,"curve_{}.png".format(saveName)))
        np.savetxt(os.path.join(save_dir,"rewards_{}.txt".format(saveName)), data[y1_index], delimiter=",")
        np.savetxt(os.path.join(save_dir,"step_{}.txt".format(saveName)), data[step_index], delimiter=",")
    if show:
        plt.show()

def load_data_and_draw(reward_txt, step_txt):
    y = np.loadtxt(reward_txt, delimiter=",")
    x = np.arange(0, len(y))
    step = np.loadtxt(step_txt, delimiter=",")
    learning_curve([x, y, step], y1_func=np.mean, step_index=None, title="Test curve", save=False, show=True)

if __name__ == '__main__':
    load_data_and_draw("../../data/models/2021_11_11_00_12_PedsMoveEnv/rewards_f81f8533-4240-11ec-84e7-3cecef04b81e.txt",
                       "../../data/models/2021_11_11_00_12_PedsMoveEnv/step_f81f8533-4240-11ec-84e7-3cecef04b81e.txt")

# while batch_size > 0:
#     index = int(random.random() * self.len)
#     episode_len = self.episodes[index].len
#     count = int(round(random.random() * episode_len))
#     count = min(count, batch_size, episode_len)
#     sample_trans += self.episodes[index].sample(count)
#     batch_size -= count
#return sample_trans