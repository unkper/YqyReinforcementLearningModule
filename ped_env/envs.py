import copy
import random
from math import inf, sqrt, pow
from numpy import fliplr, flipud

import gym
import pyglet
import numpy

from Box2D import (b2World, b2Vec2)
from typing import List, Tuple

from gym.spaces import Box, Discrete
from gym.utils import seeding

from ped_env.classes import PedsRLHandler
from ped_env.objects import BoxWall, Person, Exit
from ped_env.utils.colors import (ColorBlue, ColorWall, ColorRed)
from ped_env.event_listeners import MyContactListener
from ped_env.utils.misc import ObjectType
from ped_env.utils.maps import Map
from ped_env.functions import parse_discrete_action
from ped_env.utils.viewer import PedsMoveEnvViewer

TICKS_PER_SEC = 60
vel_iters, pos_iters = 6, 2
ACTION_DIM = 9

class Model:
    def __init__(self):
        self.world = None

    def find_person(self, person_id) -> Person:
        raise NotImplemented

    def pop_ped_from_renderlist(self, person_id):
        raise NotImplemented

class PedsMoveEnvFactory():
    def __init__(self, world: b2World, l1, l2):
        self.world = world
        self.l1 = l1
        self.l2 = l2

    def create_walls(self, start_nodes, width_height, object_type, color=ColorWall, CreateClass=BoxWall):
        if CreateClass is Exit:
            return [CreateClass(self.world, start_nodes[i][0],
                                start_nodes[i][1], start_nodes[i][2], width_height[0],
                                width_height[1], self.l1) for i in range(len(start_nodes))]
        else:
            return [CreateClass(self.world, start_nodes[i][0],
                                start_nodes[i][1], width_height[0],
                                width_height[1], self.l1, object_type, color) for i in range(len(start_nodes))]

    def create_people(self, start_nodes, exit_type, test_mode=False):
        '''
        根据start_nodes创建同等数量的行人
        :param start_nodes:
        :param exit_type:
        :return:
        '''
        return [Person(self.world, start_nodes[i][0],
                       start_nodes[i][1], exit_type, self.l1, self.l2) for i in range(len(start_nodes))]

    def random_create_persons(self, start_nodes, radius, person_num, exit_type, test_mode=False):
        start_pos = []
        for i in range(person_num):
            new_x, new_y = start_nodes[0] + radius * random.random(), start_nodes[1] + radius * random.random()
            start_pos.append((new_x, new_y))
        return self.create_people(start_pos, exit_type, test_mode)

class PedsMoveEnv(Model, gym.Env):
    viewer = None
    peds = []
    render_peds = []
    peds_exit_time = {}

    def __init__(self,
                 terrain: Map,
                 person_num=10,
                 discrete=True,
                 update_frequent=30,
                 maxStep=3000,
                 r_arrival=15,
                 r_approaching=2.5,
                 r_collision=-7.5,
                 PersonHandler=PedsRLHandler,
                 test_mode:bool=False):
        super(PedsMoveEnv, self).__init__()

        self.left_person_num = 0
        self.step_in_env = 0
        self.elements = []
        self.terrain = terrain

        self.person_num = person_num
        self.discrete = discrete
        self.maxStep = maxStep

        self.person_handler = PersonHandler(self)
        # 由PersonHandler类提供的属性代替，从而使用策略模式来加强灵活性
        self.observation_space = self.person_handler.observation_space
        self.action_space = self.person_handler.action_space
        self.agent_count = self.person_handler.agent_count
        self.distance_to_exit = []

        self.update_frequent = update_frequent
        self.r_arrival = r_arrival
        self.r_approach = r_approaching
        self.r_collision = r_collision
        # self.r_smooth = r_smooth

        #for raycast and aabb_query debug
        self.test_mode = test_mode

    def start(self, maps: numpy.ndarray, person_num_sum: int = 60, person_create_radius: float = 5):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.listener = MyContactListener(self)
        self.world.contactListener = self.listener

        self.batch = pyglet.graphics.Batch()
        self.display_level = pyglet.graphics.OrderedGroup(0)
        self.debug_level = pyglet.graphics.OrderedGroup(1)
        self.factory = PedsMoveEnvFactory(self.world,self.display_level,self.debug_level)
        # 根据shape为50*50的map来构建1*1的墙，当该处值为1代表是墙
        start_nodes_obs = []
        start_nodes_wall = []
        start_nodes_exit = []
        #按照从左往右，从上到下的遍历顺序
        for j in range(maps.shape[1]):
            for i in range(maps.shape[0]):
                if maps[i, j] == 1:
                    start_nodes_obs.append((i + 0.5, j + 0.5))
                elif maps[i, j] == 2:
                    start_nodes_wall.append((i + 0.5, j + 0.5))
                elif 9 >= maps[i, j] >= 3:
                    start_nodes_exit.append((i + 0.5, j + 0.5, maps[i, j]))
        obstacles = self.factory.create_walls(start_nodes_obs, (1, 1),  ObjectType.Obstacle, color=ColorBlue)
        exits = self.factory.create_walls(start_nodes_exit, (1, 1),  ObjectType.Exit, color=ColorRed, CreateClass=Exit)  # 创建出口
        walls = self.factory.create_walls(start_nodes_wall, (1, 1), ObjectType.Wall)  # 建造围墙

        # 随机初始化行人点，给每个生成点平均分配到不同出口的人群
        self.peds = []
        person_num_in_every_spawn = person_num_sum // len(self.terrain.start_points) \
            if person_num_sum >= len(self.terrain.start_points) else 1
        person_num = [person_num_in_every_spawn
                      for _ in range(len(self.terrain.start_points))]
        for i, num in enumerate(person_num):
            exit_type = i % len(self.terrain.exits)
            if not self.test_mode:
                self.peds.extend(self.factory.random_create_persons(self.terrain.start_points[i],
                                                                    person_create_radius, num,
                                                                    exit_type + 3, self.test_mode))  # 因为出口从3开始编号，依次给行人赋予出口编号值
            else:
                self.peds.extend(self.factory.create_people(self.terrain.start_points[i], exit_type + 3, self.test_mode))
        self.left_person_num = sum(person_num)
        self.render_peds = copy.copy(self.peds)
        self.elements = exits + obstacles + walls + self.render_peds
        # 得到一开始各个智能体距离出口的距离
        self.distance_to_exit.clear()
        self.get_ped_distance_to_exit()

    def setup_graphics(self):
        for ele in self.elements:
            ele.setup(self.batch, self.terrain.get_render_scale())

    def delete_person(self, per: Person):
        self.world.DestroyBody(per.body)
        self.pop_ped_from_renderlist(per.id)
        per.delete()
        self.left_person_num -= 1

    def pop_ped_from_renderlist(self, person_id):
        """
        将行人从渲染队列中移除
        :param person_id:
        :return:
        """
        for idx, per in enumerate(self.elements):
            if per.type == ObjectType.Agent and per.id == person_id:
                self.elements.pop(idx)
                return
        raise Exception("移除一个不存在的行人!")

    def get_ped_distance_to_exit(self):
        for ped in self.peds:
            # min_dis = self.get_nearest_exit_dis((ped.getX, ped.getY))
            dis = self.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
            self.distance_to_exit.append(dis)

    def get_ped_nearest_exit_dis(self, person_pos):
        x, y = person_pos
        min = 100
        for exit_point in self.terrain.exits:
            ex, ey = exit_point
            dis = sqrt(pow(ex - x, 2) + pow(ey - y, 2))
            min = dis if min > dis else min
        return min

    def get_ped_to_exit_dis(self, person_pos, exit_type):
        ex, ey = self.terrain.exits[exit_type - 3]  # 从3开始编号
        x, y = person_pos
        return sqrt(pow(ex - x, 2) + pow(ey - y, 2))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # reset ped_env
        Person.counter = 0
        BoxWall.counter = 0
        Exit.counter = 0
        self.step_in_env = 0
        self.peds_exit_time.clear()
        self.peds.clear()
        self.render_peds.clear()
        self.elements.clear()
        self.start(self.terrain.map, person_num_sum=self.person_num
                   , person_create_radius=self.terrain.create_radius)
        # 添加初始观察状态
        init_obs = []
        for per in self.peds:
            init_obs.append(self.person_handler.get_observation(per))
        return init_obs

    def step(self, actions):
        rewards = []
        is_done = [False for _ in range(self.agent_count)]
        obs = []
        if len(actions) != self.agent_count: raise Exception("动作向量与智能体数量不匹配!")
        if len(actions[0]) != ACTION_DIM: raise Exception("动作向量的长度不正确!")

        for i, ped in enumerate(self.peds):
            self.person_handler.set_action(ped, actions[i])

        for i in range(self.update_frequent):
            # update box2d physical world
            self.world.Step(1 / TICKS_PER_SEC, vel_iters, pos_iters)
        self.world.ClearForces()

        # 检查是否有行人到达出口要进行移除，该环境中智能体是合作关系，因此使用统一奖励为好
        for i, per in enumerate(self.peds):
            obs.append(self.person_handler.get_observation(per))
            if per.is_done and not per.has_removed:#移除到达出口的行人
                self.delete_person(per)
            reward = self.person_handler.get_reward(per, i)
            rewards.append(reward)
        if self.left_person_num == 0:
            is_done = [True for _ in range(self.agent_count)]
            # print("所有行人都已到达出口，重置环境!")
        self.step_in_env += 1
        if self.step_in_env > self.maxStep:  # 如果maxStep步都没有完全撤离，is_done直接为True
            is_done = [True for _ in range(self.agent_count)]
            # print("在{}步时强行重置环境!".format(self.maxStep))
        return obs, rewards, is_done, "PedMoveEnv"

    once = False
    def debug_step(self):
        if not self.once:
            for i,ped in enumerate(self.peds):
                print("Agent{}:".format(i))
                # ped.raycast(self.world, b2Vec2(1,0), test_mode=True)
                ped.aabb_query(self.world, 1, test_mode=True)
                print("############")
            self.once = True

    def render(self, mode="human"):
        if self.viewer is None:  # 如果调用了 render, 而且没有 viewer, 就生成一个
            self.viewer = PedsMoveEnvViewer(self)
        self.viewer.render()  # 使用 Viewer 中的 render 功能
