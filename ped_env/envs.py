import copy
import random
from math import inf

import gym
import pyglet
import numpy

from Box2D import (b2World)
from typing import List, Tuple

from gym.spaces import Box, Discrete
from gym.utils import seeding

from ped_env.objects import BoxWall, Person, Exit
from ped_env.utils.colors import (ColorBlue, ColorWall, ColorRed)
from ped_env.event_listeners import MyContactListener
from ped_env.utils.misc import ObjectType
from ped_env.utils.maps import Map, map_03
from ped_env.functions import parse_discrete_action
from ped_env.utils.viewer import PedsMoveEnvViewer

TICKS_PER_SEC = 60
vel_iters, pos_iters = 6, 2
ACTION_DIM = 9

class Model():
    def __init__(self):
        self.world = None

    def find_person(self, person_id) -> Person:
        raise NotImplemented

    def pop_person_from_renderlist(self, person_id):
        raise NotImplemented

class PedsMoveEnvFactory():
    def __init__(self, world:b2World):
        self.world = world

    def create_walls(self, start_nodes, width_height, same=True, Color = ColorWall, CreateClass = BoxWall):
        if same:
            return [CreateClass(self.world, start_nodes[i][0],
                              start_nodes[i][1], width_height[0],
                              width_height[1], Color) for i in range(len(start_nodes))]
        else:
            return [CreateClass(self.world, start_nodes[i][0],
                         start_nodes[i][1], width_height[i][0],
                         width_height[i][1], Color) for i in range(len(start_nodes))]

    def create_people(self, start_nodes):
        return [Person(self.world, start_nodes[i][0],
                            start_nodes[i][1]) for i in range(len(start_nodes))]

    def random_create_persons(self, start_nodes, radius, person_num):
        start_pos = []
        for i in range(person_num):
            new_x, new_y = start_nodes[0] + radius * random.random(), start_nodes[1] + radius * random.random()
            start_pos.append((new_x, new_y))
        return self.create_people(start_pos)


class PedsMoveEnv(Model, gym.Env):
    viewer = None
    peds = []
    render_peds = []
    peds_exit_time= {}
    def __init__(self,
                 terrain:Map,
                 person_num = 10,
                 discrete = True,
                 update_frequent = 30,
                 maxStep = 3000,
                 r_arrival = 15,
                 r_approaching = 2.5,
                 r_collision = -7.5):
        super(PedsMoveEnv, self).__init__()

        self.left_person_num = 0
        self.step_in_env = 0
        self.elements = []
        self.terrain = terrain

        self.person_num = person_num
        self.discrete = discrete
        self.maxStep = maxStep

        #强化学习MDP定义区域
        #定义观察空间为[智能体id,8个方向的传感器,智能体当前位置(x,y),智能体当前速度(dx,dy)]一共13个值
        self.observation_space = [Box( -inf, inf, (13,)) for _ in range(self.person_num)]
        #定义动作空间为[不动，向左，向右，向上，向下]施加1N的力
        self.action_space = [Discrete(ACTION_DIM) for _ in range(self.person_num)]
        self.agent_count = person_num
        self.distance_to_exit = []

        self.update_frequent = update_frequent
        self.r_arrival = r_arrival
        self.r_approach = r_approaching
        self.r_collision = r_collision
        # self.r_smooth = r_smooth

    def start(self, maps:numpy.ndarray, person_num:List=[30,30], person_create_radius:float = 5):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.factory = PedsMoveEnvFactory(self.world)
        self.listener = MyContactListener(self)
        self.world.contactListener = self.listener

        self.batch = pyglet.graphics.Batch()
        # 根据shape为50*50的map来构建1*1的墙，当该处值为1代表是墙
        start_nodes_obs = []
        start_nodes_wall = []
        start_nodes_exit = []
        for i in range(maps.shape[0]):
            for j in range(maps.shape[1]):
                if maps[j, i] == 1:
                    start_nodes_obs.append((i+0.5,j+0.5))
                elif maps[j, i] == 2:
                    start_nodes_wall.append((i+0.5,j+0.5))
                elif maps[j, i] == 3:
                    start_nodes_exit.append((i+0.5,j+0.5))
        obstacles = self.factory.create_walls(start_nodes_obs, (1, 1), Color=ColorBlue)
        exits = self.factory.create_walls(start_nodes_exit, (1, 1), Color=ColorRed, CreateClass=Exit)# 创建出口
        walls = self.factory.create_walls(start_nodes_wall, (1, 1))# 建造围墙
        # 随机初始化行人点
        self.peds = []
        for i,num in enumerate(person_num):
            self.peds.extend(self.factory.random_create_persons(self.terrain.start_points[i],
                                person_create_radius, num))
        self.left_person_num = sum(person_num)
        self.render_peds = copy.copy(self.peds)
        self.elements = exits + obstacles + walls + self.render_peds
        #得到一开始各个智能体距离出口的距离
        self.distance_to_exit.clear()
        self.get_peds_distance_to_exit()

    def setup_graphics(self):
        for ele in self.elements:
            ele.setup(self.batch)

    def delete_person(self, per: Person):
        self.world.DestroyBody(per.body)
        self.pop_person_from_renderlist(per.id)
        per.delete()
        self.left_person_num -= 1

    def pop_person_from_renderlist(self, person_id):
        '''
        将行人从渲染队列中移除
        :param person_id:
        :return:
        '''
        for idx, per in enumerate(self.elements):
            if per.type == ObjectType.Agent and per.id == person_id:
                self.elements.pop(idx)
                return
        raise Exception("移除一个不存在的行人!")

    def get_peds_distance_to_exit(self):
        for ped in self.peds:
            min_dis = self.get_nearest_exit_dis((ped.getX, ped.getY))
            self.distance_to_exit.append(min_dis)

    def get_nearest_exit_dis(self, person_pos):
        x, y = person_pos
        min = 100
        for exit_point in self.terrain.exits:
            ex, ey = exit_point
            dis = ((ex - x)**2 + (ey - y)**2)**0.5
            min = dis if min > dis else min
        return min

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        #reset ped_env
        Person.counter = 0
        BoxWall.counter = 0
        Exit.counter = 0
        self.step_in_env  = 0
        self.peds_exit_time.clear()
        self.peds.clear()
        self.render_peds.clear()
        self.elements.clear()
        self.start(self.terrain.map, person_num=[self.person_num//2, self.person_num//2]
                   ,person_create_radius = self.terrain.create_radius)

        init_obs = []
        for per in self.peds:
            init_obs.append(per.get_observation(self.world))
        return init_obs

    def step(self, actions):
        rewards = []
        is_done = [False for _ in range(self.agent_count)]
        obs = []
        if len(actions) != self.person_num: raise Exception("动作向量与智能体数量不匹配!")
        if len(actions[0]) != ACTION_DIM: raise Exception("动作向量的长度不正确!")

        for i, ped in enumerate(self.peds):
            ped.self_driven_force(parse_discrete_action(actions[i]))
            ped.fraction_force()

        for i in range(self.update_frequent):
            # update box2d physical world
            self.world.Step(1 / TICKS_PER_SEC, vel_iters, pos_iters)

        # 检查是否有行人到达出口要进行移除
        for i, per in enumerate(self.peds):
            reward = 0.0
            obs.append(per.get_observation(self.world))

            if per.collide_with_wall or per.collide_with_agent:  # 如果智能体与墙或者其他智能体相撞，奖励减一
                reward += self.r_collision

            if per.is_done and not per.has_removed:
                reward += self.r_arrival  # 智能体到达出口获得10的奖励
                self.delete_person(per)
            elif per.is_done:
                pass
            else:
                last_dis = self.distance_to_exit[i]
                now_dis = self.get_nearest_exit_dis((per.getX, per.getY))
                if last_dis != now_dis:
                    reward += self.r_approach * (last_dis - now_dis)  # 给予(之前离出口距离-目前离出口距离)的差值
                    self.distance_to_exit[i] = now_dis
                else:
                    reward += self.r_collision  # 给予停止不动的行人以碰撞惩罚
            rewards.append(reward)

        if self.left_person_num == 0:
            is_done = [True for _ in range(self.agent_count)]
            # print("所有行人都已到达出口，重置环境!")
        self.step_in_env += 1
        if self.step_in_env > self.maxStep:  # 如果maxStep步都没有完全撤离，is_done直接为True
            is_done = [True for _ in range(self.agent_count)]
            # print("在{}步时强行重置环境!".format(self.maxStep))
        return (obs, rewards, is_done, "PedMoveEnv")

    def render(self, mode="human"):
        if self.viewer is None:  # 如果调用了 render, 而且没有 viewer, 就生成一个
            self.viewer = PedsMoveEnvViewer(self)
        self.viewer.render()  # 使用 Viewer 中的 render 功能

