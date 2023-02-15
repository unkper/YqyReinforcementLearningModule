import numpy as np
import pyglet


class Generator(pyglet.window.Window):
    resize_ratio = 5
    def __init__(self, height, width):
        super().__init__(width * Generator.resize_ratio,
                         height * Generator.resize_ratio)
        self.map_arr = np.zeros([height, width], dtype=np.int)
        self.start_points = []

        self.height = height
        self.width = width

    def init_map(self):
        # 给周围添加一圈障碍物
        self.map_arr[0, :] = 1
        self.map_arr[self.height - 1, :] = 1
        self.map_arr[:, self.width - 1] = 1
        self.map_arr[:, 0] = 1
        # 启动窗口以供绘制
