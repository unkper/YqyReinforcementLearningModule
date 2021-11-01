import copy
import random
from math import inf, sqrt, pow

import gym
import pyglet
import numpy

from Box2D import (b2World, b2Vec2)
from typing import List, Tuple, Dict

from gym.utils import seeding

from ped_env.classes import PedsRLHandler, ACTION_DIM, PedsRLHandlerWithCooper
from ped_env.objects import BoxWall, Person, Exit
from ped_env.utils.colors import (ColorBlue, ColorWall, ColorRed)
from ped_env.event_listeners import MyContactListener
from ped_env.utils.misc import ObjectType
from ped_env.utils.maps import Map
from ped_env.utils.viewer import PedsMoveEnvViewer
from ped_env.functions import calculate_each_group_num

TICKS_PER_SEC = 50
vel_iters, pos_iters = 6, 2

class Model:
    def __init__(self):
        self.world = None

    def find_person(self, person_id) -> Person:
        raise NotImplemented

    def pop_ped_from_not_arrived(self, person_id):
        raise NotImplemented

class PedsMoveEnvFactory():
    GROUP_SIZE = 0.25
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

    def inner_create_persons_in_radius(self, start_node, radius, person_num, exit_type, test_mode=False):
        start_pos = []
        for i in range(person_num):
            new_x, new_y = start_node[0] + radius * random.random(), start_node[1] + radius * random.random()
            start_pos.append((new_x, new_y))
        return self.create_people(start_pos, exit_type, test_mode)

    def set_group_process(self, group, leader_dic, persons):
        leader = random.sample(group, 1)[0]  # 随机选取一人作为leader
        leader.is_leader = True
        leader_dic.update({per: leader for per in group if per != leader})  # 将leader和follow的映射关系添加
        persons.extend(group)

    def create_group_persons_in_radius(self, start_node, radius, person_num, leader_dic:Dict, group_size, exit_type, test_mode=False):
        each_group_num = calculate_each_group_num(group_size, person_num)
        persons = []
        for num in each_group_num:
            group_center_node = (start_node[0] + radius * random.random(), start_node[1] + radius * random.random())
            group = self.inner_create_persons_in_radius(group_center_node, self.GROUP_SIZE, num, exit_type, test_mode)
            self.set_group_process(group, leader_dic, persons)
        return persons

    def random_create_persons(self, terrain:Map, person_num, leader_dic, group_size, exit_type, test_mode=False):
        h, w = terrain.map.shape[0], terrain.map.shape[1]
        each_group_num = calculate_each_group_num(group_size, person_num)
        persons = []
        for num in each_group_num:
            while True:
                new_x, new_y = random.random() * w,random.random() * h
                if 1 < new_x < w and 1 < new_y < h and terrain.map[int(new_x), int(new_y)] == 0:
                    break
            group = self.inner_create_persons_in_radius((new_x, new_y), self.GROUP_SIZE, num, exit_type, test_mode)
            self.set_group_process(group, leader_dic, persons)
        return persons

class PedsMoveEnv(Model, gym.Env):
    viewer = None
    peds = []
    not_arrived_peds = []
    peds_exit_time = {}

    def __init__(self,
                 terrain: Map,
                 person_num=10,
                 group_size:Tuple=(1,6),
                 discrete=True,
                 frame_skipping=8,
                 maxStep=3000,
                 PersonHandler=PedsRLHandlerWithCooper,
                 planning_mode:bool=False,
                 test_mode:bool=False):
        '''
        一个基于Box2D和pyglet的多行人强化学习仿真环境
        对于一个有N个人的环境，其状态空间为：[o1,o2,...,oN]，每一个o都是一个长度为14的list，其代表的意义为：
        [智能体编号,8个方向的传感器值,智能体当前位置,智能体当前速度,当前步数]，动作空间为[a1,a2,...,aN]，其中a为Discrete(9)
        分别代表[不动,水平向右,斜向上,...]施加力，奖励为[r1,r2,...,rN],是否结束为[is_done1,...,is_doneN]
        :param terrain: 地图类，其中包含地形ndarray，出口，行人生成点和生成半径
        :param person_num: 要生成的行人总数
        :param discrete: 动作空间是否离散（目前必须）
        :param frame_skipping: 一次step跳过的帧数，等于一次step环境经过frame_skipping*1/TICKS_PER_SEC秒
        :param maxStep: 经过多少次step后就强行结束环境，所有行人到达终点时也会结束环境
        :param PersonHandler: 用于处理有关于行人状态空间，动作与返回奖励的类
        :param planning_mode:用于planner的规划时使用，主要区别是在全场随机生成智能体
        :param test_mode: 是否debug
        :param group_size:一个团体的人数，其中至少包含1个leader和多个follower
        '''
        super(PedsMoveEnv, self).__init__()

        self.planning_mode = planning_mode
        self.left_person_num = 0
        self.step_in_env = 0
        self.elements = []
        self.terrain = terrain

        self.person_num = person_num
        self.discrete = discrete
        self.maxStep = maxStep

        self.distance_to_exit = []
        self.points_in_last_step = []

        self.frame_skipping = frame_skipping

        self.group_size = group_size
        self.leader_follower_dic = {}

        self.person_handler = PersonHandler(self)
        # 由PersonHandler类提供的属性代替，从而使用策略模式来加强灵活性
        self.observation_space = self.person_handler.observation_space
        self.action_space = self.person_handler.action_space
        self.agent_count = self.person_handler.agent_count #agent_count是一个团队里leader的数量

        self.left_leader_num = self.agent_count

        self.col_with_agent = 0
        self.col_with_wall = 0

        #for raycast and aabb_query debug
        self.test_mode = test_mode
        self.vec = [0.0 for _ in range(self.agent_count)]

    def start(self, maps: numpy.ndarray, person_num_sum: int = 60, person_create_radius: float = 5):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        # self.listener = MyContactListener(self) #现在使用aabb_query的方式来判定
        #self.world.contactListener = self.listener

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
        self.obstacles = self.factory.create_walls(start_nodes_obs, (1, 1),  ObjectType.Obstacle, color=ColorBlue)
        self.exits = self.factory.create_walls(start_nodes_exit, (1, 1),  ObjectType.Exit, color=ColorRed, CreateClass=Exit)  # 创建出口
        self.walls = self.factory.create_walls(start_nodes_wall, (1, 1), ObjectType.Wall)  # 建造围墙

        # 随机初始化行人点，给每个生成点平均分配到不同出口的人群,并根据平均数来计算需要的领队数
        self.peds = []
        reminder = person_num_sum % len(self.terrain.start_points)
        person_num_in_every_spawn = person_num_sum // len(self.terrain.start_points) \
            if person_num_sum >= len(self.terrain.start_points) else 1
        person_num = [person_num_in_every_spawn
                      for _ in range(len(self.terrain.start_points))]
        person_num[-1] += reminder
        for i, num in enumerate(person_num):
            exit_type = i % len(self.terrain.exits)
            if not self.test_mode and not self.planning_mode:
                self.peds.extend(self.factory.create_group_persons_in_radius(self.terrain.start_points[i],
                                                                             person_create_radius,
                                                                             num,
                                                                             self.leader_follower_dic,
                                                                             self.group_size,
                                                                             exit_type + 3,
                                                                             self.test_mode))  # 因为出口从3开始编号，依次给行人赋予出口编号值
            elif not self.test_mode and self.planning_mode:
                self.peds.extend(self.factory.random_create_persons(self.terrain, num, self.leader_follower_dic, self.group_size, exit_type + 3, self.test_mode))
            else:
                self.peds.extend(self.factory.random_create_persons(self.terrain, num, self.leader_follower_dic, self.group_size, exit_type + 3, self.test_mode))
                #self.peds.extend(self.factory.create_people(self.terrain.start_points[i], exit_type + 3, self.test_mode))
        self.left_person_num = sum(person_num)
        self.left_leader_num = self.agent_count
        self.not_arrived_peds = copy.copy(self.peds)
        self.elements = self.exits + self.obstacles + self.walls + self.not_arrived_peds
        # 得到一开始各个智能体距离出口的距离
        self.distance_to_exit.clear()
        self.get_peds_distance_to_exit()

    def setup_graphics(self):
        for ele in self.elements:
            ele.setup(self.batch, self.terrain.get_render_scale())

    def delete_person(self, per: Person):
        self.world.DestroyBody(per.body)
        self.pop_ped_from_not_arrived(per.id)
        per.delete()
        self.left_person_num -= 1
        if per.is_leader: self.left_leader_num -= 1

    def pop_ped_from_not_arrived(self, person_id):
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

    def get_peds_distance_to_exit(self):
        for ped in self.peds:
            # min_dis = self.get_nearest_exit_dis((ped.getX, ped.getY))
            dis = self.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
            self.distance_to_exit.append(dis)
            self.points_in_last_step.append((ped.getX, ped.getY))

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

    def get_ped_rel_pos_to_exit(self, person_pos, exit_type):
        ex, ey = self.terrain.exits[exit_type - 3]  # 从3开始编号
        x, y = person_pos
        return (ex - x, ey - y)

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
        self.not_arrived_peds.clear()
        self.elements.clear()
        self.start(self.terrain.map, person_num_sum=self.person_num
                   , person_create_radius=self.terrain.create_radius)
        # 添加初始观察状态
        init_obs = []
        for per in self.peds:
            if per.is_leader:
                init_obs.append(self.person_handler.get_observation(per))
        return init_obs

    def step(self, actions):
        is_done = [False for _ in range(self.agent_count)]
        if len(actions) != self.agent_count: raise Exception("动作向量与智能体数量不匹配!")
        if len(actions[0]) != ACTION_DIM: raise Exception("动作向量的长度不正确!")
        #清空上一步的碰撞状态

        for i in range(self.frame_skipping):
            # update box2d physical world
            leader_index = 0
            for ped in self.peds:
                if (ped.is_done and not ped.has_removed) \
                    or (not ped.is_leader and not ped.has_removed
                    and self.leader_follower_dic[ped].is_done):  # 移除到达出口的leader和follower
                    self.delete_person(ped)
                if ped.is_leader:
                    #是leader用强化学习算法来控制
                    self.person_handler.set_action(ped, actions[leader_index])
                    leader_index += 1
                else:
                    leader = self.leader_follower_dic.get(ped)
                    #是follower用操纵力加社会力模型来控制
                    if ped.exam_leader_moved(leader.body):
                        ped.leader_follow_force(leader.body)
                        ped.fij_force(self.not_arrived_peds, self.leader_follower_dic)
                    ped.fraction_force()
                    ped.fiw_force(self.walls + self.obstacles + self.exits)
                    ped.evade_controller(leader.body)
            self.world.Step(1 / TICKS_PER_SEC, vel_iters, pos_iters)
            self.world.ClearForces()
            for ped in self.peds:
                if ped.is_done and ped.has_removed:
                    ped.x, ped.y = 0, 0
                    continue
                # 首先更新目前每个ped的坐标
                ped.x, ped.y = ped.body.position.x, ped.body.position.y
                #清空上一步的碰撞状态
                if ped.collide_with_agent:
                    self.col_with_agent += 1
                if ped.collide_with_wall:
                    self.col_with_wall += 1
                ped.collide_with_wall = ped.collide_with_agent = False
                #检查是否有行人到达出口要进行移除
                def exam_self_exit(a, b):
                    return b.exit_type == a.exit_type
                exits = ped.objects_query(self.exits, 1 + ped.radius, exam_self_exit)
                if len(exits) != 0:
                    ped.is_done = True

        # 该环境中智能体是合作关系，因此使用统一奖励为好
        obs, rewards = self.person_handler.step(self.peds)
        if self.left_leader_num == 0:
            is_done = [True for _ in range(self.agent_count)]
            # print("所有行人都已到达出口，重置环境!")
        self.step_in_env += self.frame_skipping
        if self.step_in_env > self.maxStep:  # 如果maxStep步都没有完全撤离，is_done直接为True
            is_done = [True for _ in range(self.agent_count)]
            # print("在{}步时强行重置环境!".format(self.maxStep))
        return obs, rewards, is_done, self.step_in_env

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
