import uuid
import os
import datetime
import numpy as np

import matplotlib.pyplot as plt

from rl.config import Config

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

def save_parameter_setting(dir, name, config:Config):
    save_file = open(os.path.join(dir,name+"_parameter.txt"), "w+")
    for key,value in config.__dict__.items():
        save_file.write("{}:{}\n".format(key,value))
    save_file.close()

def learning_curve(data, x_index = 0, y1_index = 1, y1_func = None, step_index = None, title = "",
                   x_name = "", y_name = "",
                   y1_legend = "", saveName = "picture",
                   save=True, save_dir="./", show = False):
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
        if step_index != None:
            np.savetxt(os.path.join(save_dir,"step_{}.txt".format(saveName)), data[step_index], delimiter=",")
    if show:
        plt.show()

def contrast_learning_curve(dir, y_range = None, y_func=np.mean, smooth_step=-1, use_multiply_result=False, title="curve"):
    fig, ax = plt.subplots()
    ys = []
    names = []
    if use_multiply_result:
        for folder in os.listdir(dir):
            pa = os.path.join(dir,folder)
            y = []
            for file in os.listdir(pa):
                reward_txt = os.path.join(pa, file)
                part_y = np.loadtxt(reward_txt, delimiter=",")
                y.append(part_y)
            y = np.mean(np.array(y), axis=0)
            ys.append(y)
            names.append(folder)
    else:
        for file in os.listdir(dir):
            reward_txt = os.path.join(dir, file)
            y = np.loadtxt(reward_txt, delimiter=",")
            ys.append(y)
            names.append(file)

    for y, name in zip(ys, names):
        if y_range != None:
            y = y[y_range[0]:y_range[1]]
        if y_func != None:
            temp_y = np.zeros(y.shape[0])
            for i in range(y.shape[0]):
                temp_y[i] = y_func(y[i, :])
            y = temp_y
            if smooth_step != -1:
                temp_y = np.zeros(y.shape[0])
                for i in range(smooth_step//2, y.shape[0] - smooth_step//2):
                    temp_y[i - smooth_step//2] = np.mean(y[i - smooth_step//2:i + smooth_step//2], axis=0)

                y = temp_y[:-smooth_step]
        x = np.arange(0, len(y))
        ax.plot(x.data, y.data, label=name)
    plt.legend()
    plt.show()

def load_data_and_draw(reward_txt, step_txt):
    y = np.loadtxt(reward_txt, delimiter=",")
    x = np.arange(0, len(y))
    step = np.loadtxt(step_txt, delimiter=",")
    learning_curve([x, y, step], y1_func=np.mean, step_index=None, title="Test curve", save=False, show=True)

if __name__ == '__main__':
    # arr = np.random.random([256,101])
    # print(arr[np.unravel_index(np.argmax(arr, axis=None), arr.shape)[0], :])
    #print(os.listdir("../"))
    contrast_learning_curve("data/stats/20211224/1", y_range=(0, 800), smooth_step=20, y_func=np.mean, use_multiply_result=False)
