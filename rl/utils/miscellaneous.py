import uuid

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

def learning_curve(data, x_index = 0, y1_index = 1, y2_index = None, title = "",
                   x_name = "", y_name = "",
                   y1_legend = "", y2_legend="", saveName = "picture",
                   save=True, show=False):
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
    x = data[x_index]
    y1 = data[y1_index]
    ax.plot(x, y1, label = y1_legend)
    if y2_index is not None:
        ax.plot(x, data[y2_index], label = y2_legend)
    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='black', labelsize='medium', width=1)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend("rewards")
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    #plt.grid(True)
    import os
    import datetime
    if save:
        plt.savefig(os.path.join("./","curve_{}.png".format(saveName)))
    if show:
        plt.show()
