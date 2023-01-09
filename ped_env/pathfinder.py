import random
import time

import gym
import kdtree

from typing import List, Tuple
from collections import defaultdict

from tqdm import tqdm

import ped_env.envs
from ped_env.utils.maps import *


# https://github.com/lc6chang/Social_Force_Model
class Node:
    def __init__(self):
        #  初始化各个坐标点的g值、h值、f值、父节点
        self.g = 0
        self.h = 0
        self.f = 0
        self.father = (0, 0)


class Path:
    def __init__(self, s_pos, e_pos, path: list):
        self.start_pos = s_pos
        self.end_pos = e_pos
        self.path = path
        self.vec_dir = None

    def calculate_vec_dir_in_path(self):
        self.vec_dir = {}
        node_arr = self.path
        node_arr.append(self.end_pos)
        for i in range(len(node_arr) - 1):
            po1, po2 = node_arr[i], node_arr[i + 1]
            po1 = (int(po1[0]), int(po1[1]))
            po2 = (int(po2[0]), int(po2[1]))
            dir = (po2[0] - po1[0], po2[1] - po1[1])
            self.vec_dir[po1] = dir
        return self.vec_dir


class AStar:
    def __init__(self, map: Map):
        self.map = map
        self.barrier_list = []
        self.init_barrier_list()
        self.dir_vector_matrix_dic = dict()  # 值是出口坐标(x,y)，键是ndarray
        self.path_matrix_dic = defaultdict(dict)  # 键是出口坐标(x,y),值是一个字典(键是起始坐标(sx,sy),值是路径Path)

    def init_barrier_list(self):
        terrain = self.map.map
        for j in range(terrain.shape[1]):
            for i in range(terrain.shape[0]):
                if terrain[i, j] in (1, 2):
                    self.barrier_list.append((i, j))

    def next_loc(self, x, y, dest_x, dest_y) -> Tuple[Tuple, Path]:
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

        find_way = False
        open_list.append(start_loc)  # 起始点添加至打开列表
        # 开始算法的循环
        while True:
            if len(open_list) == 0:
                break
            now_loc = open_list[0]
            if now_loc[0] < 0 or now_loc[0] >= terrain.shape[0] or now_loc[1] < 0 or now_loc[1] >= terrain.shape[1]:
                break
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
                if temp_loc[0] < 0 or temp_loc[0] >= terrain.shape[0] or temp_loc[1] < 0 or temp_loc[1] >= \
                        terrain.shape[1]:
                    continue
                if temp_loc in self.barrier_list:  # 如果在障碍列表，则跳过
                    continue
                if temp_loc in close_list:  # 如果在关闭列表，则跳过
                    continue

                #  该节点不在open列表，添加，并计算出各种值
                if temp_loc not in open_list:
                    open_list.append(temp_loc)
                    node_matrix[temp_loc[0]][temp_loc[1]].g = (node_matrix[now_loc[0]][now_loc[1]].g +
                                                               int(((temp[0] ** 2 + temp[1] ** 2) * 100) ** 0.5))
                    node_matrix[temp_loc[0]][temp_loc[1]].h = (abs(aim_loc[0][0] - temp_loc[0])
                                                               + abs(aim_loc[0][1] - temp_loc[1])) * 10
                    node_matrix[temp_loc[0]][temp_loc[1]].f = (node_matrix[temp_loc[0]][temp_loc[1]].g +
                                                               node_matrix[temp_loc[0]][temp_loc[1]].h)
                    node_matrix[temp_loc[0]][temp_loc[1]].father = now_loc
                    continue

                #  如果在open列表中，比较，重新计算
                if node_matrix[temp_loc[0]][temp_loc[1]].g > (node_matrix[now_loc[0]][now_loc[1]].g +
                                                              int(((temp[0] ** 2 + temp[1] ** 2) * 100) ** 0.5)):
                    node_matrix[temp_loc[0]][temp_loc[1]].g = (node_matrix[now_loc[0]][now_loc[1]].g +
                                                               int(((temp[0] ** 2 + temp[1] ** 2) * 100) ** 0.5))
                    node_matrix[temp_loc[0]][temp_loc[1]].father = now_loc
                    node_matrix[temp_loc[0]][temp_loc[1]].f = (node_matrix[temp_loc[0]][temp_loc[1]].g +
                                                               node_matrix[temp_loc[0]][temp_loc[1]].h)

            #  判断是否停止
            if aim_loc[0] in close_list:
                find_way = True
                break

        if find_way:
            #  依次遍历父节点，找到下一个位置
            temp = aim_loc[0]
            path_arr = []
            while node_matrix[temp[0]][temp[1]].father != start_loc:
                temp = node_matrix[temp[0]][temp[1]].father
                _temp = (temp[0] + 0.5, temp[1] + 0.5)  # 这里加0.5是为了消除int带来的向下取整效果
                path_arr.insert(0, _temp)
            start_loc_tmp = (start_loc[0] + 0.5, start_loc[1] + 0.5)
            path_arr.insert(0, start_loc_tmp)
            #  返回下一个位置的方向向量，例如：（-1,0），（-1,1）......
            #  保存路径数据到Path类中返回
            path = Path(start_loc, aim_loc[0], path_arr)
            re = (temp[0] - start_loc[0], temp[1] - start_loc[1])
            return re, path
        else:
            print("Warning,A* find no path from {} to {}!!!".format(start_loc, aim_loc[0]))
            return (0, 0), None

    def calculate_dir_vector(self):
        terrain = self.map.map
        for exit in self.map.exits:
            vector_matrix = [[0 for i in range(terrain.shape[1])] for i in range(terrain.shape[0])]
            # 在一开始就计算好各个位置的下一步方向向量，并存储到矩阵中，以节省算力
            for j in range(terrain.shape[1]):
                for i in range(terrain.shape[0]):
                    if terrain[i, j] == 0:  # 空地
                        start_pos = (i, j)
                        end_pos = (exit[0], exit[1])
                        vector_matrix[i][j], pa = self.next_loc(i, j, int(exit[0]), int(exit[1]))
                        self.path_matrix_dic[end_pos][start_pos] = pa
            self.dir_vector_matrix_dic[exit] = vector_matrix

    def print_dir_vector_map(self, matrix: List[List]):
        terrain = self.map.map
        print_string = ""
        for j in range(terrain.shape[1]):
            for i in range(terrain.shape[0]):
                if matrix[i][j] == 0:  # 没有方向向量
                    if terrain[i][j] in (1, 2):
                        print_string += "▇"
                    elif 9 >= terrain[i][j] >= 3:
                        print_string += "$"
                    else:
                        print_string += "%"
                else:
                    vector_char = {
                        (0, 0): "E",
                        (-1, 0): "←",
                        (0, -1): "↑",
                        (0, 1): "↓",
                        (1, 0): "→",
                        (-1, 1): "↙",
                        (1, -1): "↗",
                        (1, 1): "↘",
                        (-1, -1): "↖"
                    }
                    print_string += vector_char[matrix[i][j]]
            print_string += "\n"
        return print_string


ACTION_DIM = 9

class AStarController(gym.Env):
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

    def __init__(self, env, random_policy=False, discrete=True):
        '''
        利用了行人模拟环境，并使用AStar算法做自驱动力来控制行人的行走
        :param env:
        '''
        self.env = env
        self.planner = AStar(env.terrain)
        self.exit_tree = kdtree.create(env.terrain.exits)
        self.planner.calculate_dir_vector()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.agent_count = self.env.agent_count
        self.random_policy = random_policy
        self.discrete = discrete
        if random_policy:
            self.action_space = range(ACTION_DIM) if discrete else self.action_space

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        self.env.close()

    def reset(self):
        return self.env.reset()

    def step(self, obs):
        actions = []
        for idx in range(len(obs)):
            ped = self.env.leaders[idx]
            pos_integer = [int(ped.getX), int(ped.getY)]
            exit = self.env.terrain.exits[ped.exit_type - 3]  # 根据智能体的id得到智能体要去的出口,3是因为出口从3开始编号
            dir = self.planner.dir_vector_matrix_dic[exit][pos_integer[0]][pos_integer[1]]
            if not self.random_policy:
                action = np.zeros([ACTION_DIM])
                if dir != 0:
                    action[self.vec_to_discrete_action_dic[dir]] = 1.0
            else:
                if self.discrete:
                    action = np.zeros([ACTION_DIM])
                    choose_action = random.sample(self.action_space, 1)[0]
                    action[choose_action] = 1.0
                else:
                    action = self.action_space[idx].sample()
            actions.append(action)
        next_obs, reward, is_done, info = self.env.step(actions, planning_mode=False)
        # self.env.render()
        return next_obs, reward, is_done, actions

    def play(self, episodes, render=True):
        '''
        进行episodes次行人环境的模拟
        :param episodes:
        :param render:
        :return:
        '''
        for epoch in tqdm(range(episodes)):
            step, starttime = 0, time.time()
            obs = self.env.reset()
            total_reward = 0.0
            is_done = [False]
            while not is_done[0]:
                next_obs, reward, is_done, actions = self.step(obs)
                total_reward += np.mean(reward)
                obs = next_obs
                step += 1  # self.env.frame_skipping
                if render:
                    self.env.render()
            endtime = time.time()
            print("奖励为{}!".format(total_reward))
            # print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
            #      .format(self.env.listener.col_with_agent, self.env.listener.col_with_wall))
            print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime,
                                                                                 step / (endtime - starttime)))


class AStarPolicy():
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

    def __init__(self, terrain):
        self.map = terrain.map
        self.planner = AStar(terrain)
        self.exit_tree = kdtree.create(terrain.exits)
        self.planner.calculate_dir_vector()

    def is_pos_vaild(self, pos):
        pos_x, pos_y = pos
        if pos_x < 0 or pos_x >= self.map.shape[0] or pos_y < 0 or pos_y >= self.map.shape[1]:
            return False
        if self.map[pos_x, pos_y] != 0:
            return False
        return True

    def step(self, obs):
        actions = []
        for ob in obs:
            pos_x, pos_y = ob[0:2]  # 得到智能体当前位置
            pos_integer = [int(pos_x), int(pos_y)]
            rx, ry = ob[4:6]  # 得到相对于出口的距离
            ex = pos_x + rx
            ey = pos_y + ry
            exit = self.exit_tree.search_nn((ex, ey))[0].data  # 寻找相对最近的节点
            if self.is_pos_vaild(pos_integer):
                dir = self.planner.dir_vector_matrix_dic[exit][pos_integer[0]][pos_integer[1]]
            else:
                dir = 0
            action = np.zeros([ACTION_DIM])
            if dir != 0:
                action[self.vec_to_discrete_action_dic[dir]] = 1.0
            actions.append(action)
        return actions


def recoder_for_debug(*obj):
    pass


if __name__ == '__main__':
    env = ped_env.envs.PedsMoveEnv(map_12, 32, (4, 4), random_init_mode=True)
    planner = AStarController(env)
    planner.play(20)
