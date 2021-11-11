import time
import numpy as np

from typing import List

from ped_env.utils.maps import Map, map_05, map_06, map_hard_obj
from ped_env.functions import parse_discrete_action
from ped_env.envs import PedsMoveEnv, ACTION_DIM

#https://github.com/lc6chang/Social_Force_Model
class Node:
    def __init__(self):
        #  初始化各个坐标点的g值、h值、f值、父节点
        self.g = 0
        self.h = 0
        self.f = 0
        self.father = (0, 0)

class AStar:
    def __init__(self, map:Map):
        self.map = map
        self.barrier_list = []
        self.init_barrier_list()
        self.dir_vector_matrix_dic = dict() #值是出口坐标(x,y)，键是ndarray
        self.calculate_dir_vector()

    def init_barrier_list(self):
        terrain = self.map.map
        for j in range(terrain.shape[1]):
            for i in range(terrain.shape[0]):
                if terrain[i, j] in (1, 2):
                    self.barrier_list.append((i, j))

    def next_loc(self, x, y, dest_x, dest_y):
        # 初始化各种状态
        start_loc = (x, y)  # 初始化起始点
        aim_loc = [(dest_x, dest_y)]  # 初始化目标地点
        open_list = []  # 初始化打开列表
        close_list = []  # 初始化关闭列表

        terrain = self.map.map
        # 创建存储节点的矩阵
        node_matrix = [[0 for i in range(terrain.shape[1])] for i in range(terrain.shape[0])]
        for i in range(0, terrain.shape[0]):
            for j in range(0, terrain.shape[1]):
                node_matrix[i][j] = Node()

        open_list.append(start_loc)  # 起始点添加至打开列表
        # 开始算法的循环
        while True:
            now_loc = open_list[0]
            for i in range(1, len(open_list)):  # （1）获取f值最小的点
                if node_matrix[open_list[i][0]][open_list[i][1]].f < node_matrix[now_loc[0]][now_loc[1]].f:
                    now_loc = open_list[i]
            #   （2）切换到关闭列表
            open_list.remove(now_loc)
            close_list.append(now_loc)
            #  （3）对相邻格中的每一个
            list_offset = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)]
            for temp in list_offset:
                temp_loc = (now_loc[0] + temp[0], now_loc[1] + temp[1])
                if temp_loc[0] < 0 or temp_loc[0] >= terrain.shape[0] or temp_loc[1] < 0 or temp_loc[1] >= terrain.shape[1]:
                    continue
                if temp_loc in self.barrier_list:  # 如果在障碍列表，则跳过
                    continue
                if temp_loc in close_list:  # 如果在关闭列表，则跳过
                    continue

                #  该节点不在open列表，添加，并计算出各种值
                if temp_loc not in open_list:
                    open_list.append(temp_loc)
                    node_matrix[temp_loc[0]][temp_loc[1]].g = (node_matrix[now_loc[0]][now_loc[1]].g +
                                                             int(((temp[0]**2 + temp[1]**2)*100)**0.5))
                    node_matrix[temp_loc[0]][temp_loc[1]].h = (abs(aim_loc[0][0]-temp_loc[0])
                                                               + abs(aim_loc[0][1]-temp_loc[1]))*10
                    node_matrix[temp_loc[0]][temp_loc[1]].f = (node_matrix[temp_loc[0]][temp_loc[1]].g +
                                                               node_matrix[temp_loc[0]][temp_loc[1]].h)
                    node_matrix[temp_loc[0]][temp_loc[1]].father = now_loc
                    continue

                #  如果在open列表中，比较，重新计算
                if node_matrix[temp_loc[0]][temp_loc[1]].g > (node_matrix[now_loc[0]][now_loc[1]].g +
                                                             int(((temp[0]**2+temp[1]**2)*100)**0.5)):
                    node_matrix[temp_loc[0]][temp_loc[1]].g = (node_matrix[now_loc[0]][now_loc[1]].g +
                                                             int(((temp[0]**2+temp[1]**2)*100)**0.5))
                    node_matrix[temp_loc[0]][temp_loc[1]].father = now_loc
                    node_matrix[temp_loc[0]][temp_loc[1]].f = (node_matrix[temp_loc[0]][temp_loc[1]].g +
                                                               node_matrix[temp_loc[0]][temp_loc[1]].h)

            #  判断是否停止
            if aim_loc[0] in close_list:
                break

        #  依次遍历父节点，找到下一个位置
        temp = aim_loc[0]
        while node_matrix[temp[0]][temp[1]].father != start_loc:
            temp = node_matrix[temp[0]][temp[1]].father
        #  返回下一个位置的方向向量，例如：（-1,0），（-1,1）......
        re = (temp[0] - start_loc[0], temp[1] - start_loc[1])
        return re

    def calculate_dir_vector(self):
        terrain = self.map.map
        for exit in self.map.exits:
            vector_matrix = [[0 for i in range(terrain.shape[1])] for i in range(terrain.shape[0])]
            # 在一开始就计算好各个位置的下一步方向向量，并存储到矩阵中，以节省算力
            for j in range(terrain.shape[1]):
                for i in range(terrain.shape[0]):
                    if terrain[i, j] == 0:#空地
                        vector_matrix[i][j] = self.next_loc(i, j, int(exit[0]), int(exit[1]))
            self.dir_vector_matrix_dic[exit] = vector_matrix

    def print_dir_vector_map(self, matrix:List[List]):
        terrain = self.map.map
        print_string = ""
        for j in range(terrain.shape[1]):
            for i in range(terrain.shape[0]):
                if matrix[i][j] == 0: #没有方向向量
                    if terrain[i][j] in (1,2):
                        print_string += "▇"
                    elif 9 >= terrain[i][j] >= 3:
                        print_string += "$"
                    else:
                        print_string += "%"
                else:
                    vector_char = {
                        (-1, 0):"←",
                        (0, -1):"↑",
                        (0, 1):"↓",
                        (1, 0):"→",
                        (-1, 1):"↙",
                        (1, -1):"↗",
                        (1, 1):"↘",
                        (-1, -1):"↖"
                    }
                    print_string += vector_char[matrix[i][j]]
            print_string += "\n"
        return print_string

class AStarController():
    vec_to_discrete_action_dic = {
        (0, 0): 0,
        (1, 0): 1,
        (1, 1): 2,
        (0, 1): 3,
        (-1, 1): 4,
        (-1, 0): 5,
        (-1, -1): 6,
        (0, -1): 7,
        (1, -1): 8
    }
    def __init__(self, env:PedsMoveEnv, recorder=None):
        '''
        利用了行人模拟环境，并使用AStar算法做自驱动力来控制行人的行走
        :param env:
        '''
        self.env = env
        self.planner = AStar(env.terrain)
        self.recorder = recorder

    def step(self, obs):
        actions = []
        for idx in range(len(obs)):
            ped = self.env.leaders[idx]
            pos_integer = [int(ped.getX), int(ped.getY)]
            exit = self.env.terrain.exits[ped.exit_type - 3] #根据智能体的id得到智能体要去的出口,3是因为出口从3开始编号
            dir = self.planner.dir_vector_matrix_dic[exit][pos_integer[0]][pos_integer[1]]
            action = np.zeros([ACTION_DIM])
            if dir != 0:
                action[self.vec_to_discrete_action_dic[dir]] = 1.0
            actions.append(action)
        return actions

    def play(self, episodes, render=True):
        '''
        进行episodes次行人环境的模拟
        :param episodes:
        :param render:
        :return:
        '''
        for epoch in range(episodes):
            step, starttime = 0, time.time()
            obs = self.env.reset()
            is_done = [False]
            while not is_done[0]:
                action = self.step(obs)
                next_obs, reward, is_done, info = self.env.step(action)
                self.recorder(obs, action, reward, is_done, next_obs)
                obs = next_obs
                step += self.env.frame_skipping
                if render:
                    self.env.render()
            endtime = time.time()
            #print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
            #      .format(self.env.listener.col_with_agent, self.env.listener.col_with_wall))
            print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime, step / (endtime - starttime)))

def recoder_for_debug(*obj):
    pass

def test_func01():
    m = map_05
    start_time = time.time()
    planner = AStar(m)
    file = open("answer_{}.txt".format(m.name), "w+")
    for exit in m.exits:
        file.write(str(exit) + ":\n")
        file.write(planner.print_dir_vector_map(planner.dir_vector_matrix_dic[exit]))
        file.write("\n\n")
    print(time.time() - start_time)

if __name__ == '__main__':
    env = PedsMoveEnv(map_05, person_num=30, group_size=(1,6), frame_skipping=8, maxStep=2000, planning_mode=True)
    controller = AStarController(env, recorder=recoder_for_debug)
    controller.play(5, True)
