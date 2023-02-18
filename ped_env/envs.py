import copy
import math
import random

import gym
import pyglet
import numpy as np
import logging

from math import sqrt, pow

from Box2D import (b2World, b2Vec2)
from typing import Tuple, Dict, cast, List, Optional, Any, Type
from collections import defaultdict

from gym import Space
from numba import njit

from ped_env.mdp import PedsRLHandlerWithoutForce
from ped_env.pathfinder import AStar
from ped_env.listener import MyContactListener
from ped_env.objects import BoxWall, Person, Exit, Group
from ped_env.utils.colors import (ColorBlue, ColorWall, ColorRed)
from ped_env.utils.misc import ObjectType
from ped_env.utils.maps import Map
from ped_env.utils.viewer import PedsMoveEnvViewer
from ped_env.functions import calculate_each_group_num, calculate_groups_person_num
from ped_env.settings import TICKS_PER_SEC, vel_iters, pos_iters, ACTION_DIM, GROUP_SIZE


class Spawner:

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

    def create_wall_extra(self, wall_info, color=ColorWall):
        walls = []
        for info in wall_info:
            if len(info) == 2:
                start_node, type = info
            else:
                start_node, next_start_node, type = info
            if type == 'box':
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.BOX_WALL_WIDTH, BoxWall.BOX_WALL_HEIGHT,
                                     self.l1, ObjectType.Wall, color))
            elif type == "lwall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_WIDTH, BoxWall.PIECE_WALL_HEIGHT,
                                     self.l1, ObjectType.Wall, color))
            elif type == "rwall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_WIDTH, BoxWall.PIECE_WALL_HEIGHT,
                                     self.l1, ObjectType.Wall, color))
            elif type == "uwall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_HEIGHT, BoxWall.PIECE_WALL_WIDTH,
                                     self.l1, ObjectType.Wall, color))
            elif type == "dwall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_HEIGHT, BoxWall.PIECE_WALL_WIDTH,
                                     self.l1, ObjectType.Wall, color))
            elif type == "midrow_wall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_HEIGHT, BoxWall.PIECE_WALL_WIDTH,
                                     self.l1, ObjectType.Wall, color))
            elif type == "midcolumn_wall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_WIDTH, BoxWall.PIECE_WALL_HEIGHT,
                                     self.l1, ObjectType.Wall, color))
            elif type == "corner_left_up_wall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_WIDTH, BoxWall.PIECE_WALL_HEIGHT,
                                     self.l1, ObjectType.Wall, color))
                walls.append(BoxWall(self.world, next_start_node[0], next_start_node[1],
                                     BoxWall.PIECE_WALL_HEIGHT, BoxWall.PIECE_WALL_WIDTH,
                                     self.l1, ObjectType.Wall, color))
            elif type == "corner_left_down_wall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_WIDTH, BoxWall.PIECE_WALL_HEIGHT,
                                     self.l1, ObjectType.Wall, color))
                walls.append(BoxWall(self.world, next_start_node[0], next_start_node[1],
                                     BoxWall.PIECE_WALL_HEIGHT, BoxWall.PIECE_WALL_WIDTH,
                                     self.l1, ObjectType.Wall, color))
            elif type == "corner_right_up_wall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_WIDTH, BoxWall.PIECE_WALL_HEIGHT,
                                     self.l1, ObjectType.Wall, color))
                walls.append(BoxWall(self.world, next_start_node[0], next_start_node[1],
                                     BoxWall.PIECE_WALL_HEIGHT, BoxWall.PIECE_WALL_WIDTH,
                                     self.l1, ObjectType.Wall, color))
            elif type == "corner_right_down_wall":
                walls.append(BoxWall(self.world, start_node[0], start_node[1],
                                     BoxWall.PIECE_WALL_WIDTH, BoxWall.PIECE_WALL_HEIGHT,
                                     self.l1, ObjectType.Wall, color))
                walls.append(BoxWall(self.world, next_start_node[0], next_start_node[1],
                                     BoxWall.PIECE_WALL_HEIGHT, BoxWall.PIECE_WALL_WIDTH,
                                     self.l1, ObjectType.Wall, color))
        return walls

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

    def create_group_persons_in_radius(self, terrain: Map, idx, person_num, groups, group_dic: Dict, group_size,
                                       test_mode=False):
        each_group_num = calculate_each_group_num(group_size, person_num)
        persons = []
        for num in each_group_num:
            group_center_node = (terrain.start_points[idx][0] + terrain.create_radius * random.random(),
                                 terrain.start_points[idx][1] + terrain.create_radius * random.random())
            group = self.inner_create_persons_in_radius(group_center_node, GROUP_SIZE, num,
                                                        terrain.get_random_exit(idx), test_mode)
            Group.set_group_process(group, groups, group_dic, persons)
        return persons

    def random_create_persons(self, terrain: Map, idx, person_num, groups, group_dic, group_size, start_point_dic,
                              test_mode=False):
        # h, w = terrain.map.shape[0], terrain.map.shape[1]
        each_group_num = calculate_each_group_num(group_size, person_num)
        persons = []
        for num in each_group_num:
            # while True:
            #     new_x, new_y = random.random() * w,random.random() * h
            #     if 1 < new_x < w and 1 < new_y < h and terrain.map[int(new_x), int(new_y)] == 0:
            #         new_x = int(new_x) + 0.5
            #         new_y = int(new_y) + 0.5 #设置随机点为方格的正中心
            #         break
            exit_type = terrain.get_random_exit(idx)
            new_x, new_y = random.sample(start_point_dic[exit_type], 1)[0]
            group = self.inner_create_persons_in_radius((new_x, new_y), GROUP_SIZE, num, exit_type, test_mode)
            Group.set_group_process(group, groups, group_dic, persons)
        return persons


class Parser():
    def __init__(self):
        self.start_point_dic = defaultdict(list)
        self.start_nodes_wall = []
        self.start_nodes_exit = []
        self.start_nodes_obs = []

    def parse_and_create(self, map, spawn_map):
        inc = BoxWall.PIECE_WALL_WIDTH / 2
        exit_symbol = set(str(ele) for ele in range(3, 10))
        # 按照从左往右，从上到下的遍历顺序
        for j in range(map.shape[1]):
            for i in range(map.shape[0]):
                if map[i, j] == '1':
                    self.start_nodes_obs.append(((i + 0.5, j + 0.5), "box"))
                elif map[i, j] == '2':
                    self.start_nodes_wall.append((i + 0.5, j + 0.5))
                elif map[i, j] in exit_symbol:
                    self.start_nodes_exit.append((i + 0.5, j + 0.5, int(map[i, j])))
                elif map[i, j] == 'lw':
                    self.start_nodes_obs.append(((i + inc, j + 0.5), "lwall"))
                elif map[i, j] == 'rw':
                    self.start_nodes_obs.append(((i + 1 - inc, j + 0.5), "rwall"))
                elif map[i, j] == 'uw':
                    self.start_nodes_obs.append(((i + 0.5, j + 1 - inc), "uwall"))
                elif map[i, j] == 'dw':
                    self.start_nodes_obs.append(((i + 0.5, j + inc), "dwall"))
                elif map[i, j] == 'mrw':
                    self.start_nodes_obs.append(((i + 0.5, j + 0.5), "midrow_wall"))
                elif map[i, j] == 'mcw':
                    self.start_nodes_obs.append(((i + 0.5, j + 0.5), "midcolumn_wall"))
                elif map[i, j] == 'cluw':
                    self.start_nodes_obs.append(((i + inc, j + 0.5), (i + 0.5, j + 1 - inc), "corner_left_up_wall"))
                elif map[i, j] == 'cldw':
                    self.start_nodes_obs.append(((i + inc, j + 0.5), (i + 0.5, j + inc), "corner_left_down_wall"))
                elif map[i, j] == 'cruw':
                    self.start_nodes_obs.append(
                        ((i + 1 - inc, j + 0.5), (i + 0.5, j + 1 - inc), "corner_right_up_wall"))
                elif map[i, j] == 'crdw':
                    self.start_nodes_obs.append(((i + 1 - inc, j + 0.5), (i + 0.5, j + inc), "corner_right_down_wall"))

        self.start_point_dic = defaultdict(list)
        for j in range(spawn_map.shape[1]):
            for i in range(spawn_map.shape[0]):
                if 9 >= spawn_map[i, j] >= 3:
                    self.start_point_dic[spawn_map[i, j]].append((i + 0.5, j + 0.5))


class PedsMoveEnv(gym.Env):
    viewer = None
    peds = []
    not_arrived_peds = []
    once = False

    def __init__(self,
                 terrain: Map,
                 person_num=10,
                 group_size: Tuple = (1, 1),
                 discrete=True,
                 frame_skipping=8,
                 maxStep=10000,
                 person_handler = PedsRLHandlerWithoutForce,
                 disable_reward=False,
                 use_planner=False,
                 random_init_mode: bool = True,
                 train_mode: bool = True,
                 debug_mode: bool = False):
        """
        一个基于Box2D和pyglet的多行人强化学习仿真环境
        对于一个有N个人的环境，其状态空间为：[o1,o2,...,oN]，每一个o都是一个长度为14的list，其代表的意义为：
        [智能体编号,8个方向的传感器值,智能体当前位置,智能体当前速度,当前步数]，动作空间为[a1,a2,...,aN]，其中a为Discrete(9)
        分别代表[不动,水平向右,斜向上,...]施加力，奖励为[r1,r2,...,rN],是否结束为[is_done1,...,is_doneN]

        注意：由于编码上还存在问题，因此要使用整数倍的人数，及group_size*exit_num个人
        :param terrain: 地图类，其中包含地形ndarray，出口，行人生成点和生成半径
        :param person_num: 要生成的行人总数
        :param discrete: 动作空间是否离散，连续时针对每一个智能体必须输入一个二维单位方向向量（注意！）
        :param frame_skipping: 一次step跳过的帧数，等于一次step环境经过frame_skipping * 1 / TICKS_PER_SEC(50)秒
        :param maxStep: 经过多少次step后就强行结束环境，所有行人到达终点时也会结束环境
        :param person_handler: 用于处理有关于行人状态空间，动作与返回奖励的类
        :param random_init_mode: 用于planner的规划时使用，主要区别是在全场随机生成智能体
        :param train_mode: 当为False时，直到所有行人到达出口才会重置环境，当为True时，一旦所有leader到达出口才会重置环境
        :param debug_mode: 是否debug
        :param group_size:一个团体的人数，其中至少包含1个leader和多个follower
        """
        super(PedsMoveEnv, self).__init__()

        self.random_init_mode = random_init_mode
        self.left_person_num = 0
        self.step_in_env = 0
        self.elements = []
        self.terrain = terrain

        self.person_num = person_num
        self.discrete = discrete
        self.maxStep = maxStep

        self.distance_to_exit = []
        self.points_in_last_step = []
        self.init_map_points = False

        self.frame_skipping = frame_skipping
        self.group_size = group_size
        self.person_handler = person_handler(self, use_planner=use_planner)
        # 由PersonHandler类提供的属性代替，从而使用策略模式来加强灵活性
        # self.observation_space = self.person_handler.observation_space[0]
        # self.action_space = self.person_handler.action_space[0]
        self.agent_count = self.person_handler.agent_count  # agent_count是一个团队里leader的数量

        self.left_leader_num = self.agent_count

        self.collision_between_agents = 0
        self.collide_wall = 0

        # for raycast and aabb_query debug
        self.train_mode = train_mode
        self.debug_mode = debug_mode
        self.vec = [0.0 for _ in range(self.agent_count)]

        self.path_finder = AStar(self.terrain)

        # for pettingzoo interface
        self._cumulative_rewards = defaultdict(int)
        # self.agents = ["agent_{}".format(i) for i in range(self.agent_count)]
        self.agents = [str(i) for i in range(self.agent_count)]
        self.possible_agents = copy.deepcopy(self.agents)
        self.max_num_agents = len(self.possible_agents)
        self.observation_spaces = {agentid: copy.copy(self.person_handler.observation_space[0]) for agentid in
                                   self.agents}
        self.action_spaces = {agentid: copy.copy(self.person_handler.action_space[0]) for agentid in self.agents}

        self.agents_dict = {}
        self.agents_rev_dict = {}

        self.disable_reward = disable_reward
        if disable_reward:
            logging.warning(u"当前环境将使用无奖励机制!")

    @property
    def num_agents(self):
        return len(self.agents)

    def reset_property(self):
        # reset ped_env,清空所有全局变量以供下一次使用
        Person.counter = 0
        BoxWall.counter = 0
        Exit.counter = 0
        Group.counter = 0
        self.collide_wall = self.collision_between_agents = 0
        self.step_in_env = 0
        self.peds.clear()
        self.not_arrived_peds.clear()
        self.elements.clear()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def initialize_env(self, maps: np.ndarray, spawn_maps: np.ndarray, person_num_sum: int = 60):
        # 创建物理引擎
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.listener = MyContactListener(self)  # 现在使用aabb_query的方式来判定
        self.world.contactListener = self.listener
        # 创建渲染所需
        self.batch = pyglet.graphics.Batch()
        self.render_element_dict = {}
        self.display_level = pyglet.graphics.OrderedGroup(0)
        self.debug_level = pyglet.graphics.OrderedGroup(1)
        # 创建一个行人工厂以供生成行人调用
        self.factory = Spawner(self.world, self.display_level, self.debug_level)
        # 是否按照地图生成墙
        if not self.init_map_points:
            # 根据shape为50*50的map来构建1*1的墙，当该处值为1代表是墙
            self.parser = Parser()
            self.parser.parse_and_create(maps, spawn_maps)
            self.init_map_points = True

        self.obstacles = self.factory.create_wall_extra(self.parser.start_nodes_obs, color=ColorBlue)
        self.exits = self.factory.create_walls(self.parser.start_nodes_exit, (1, 1), ObjectType.Exit, color=ColorRed,
                                               CreateClass=Exit)  # 创建出口
        self.walls = self.factory.create_walls(self.parser.start_nodes_wall, (1, 1), ObjectType.Wall, ColorWall)  # 建造围墙

        # 随机初始化行人点，给每个生成点平均分配到不同出口的人群,并根据平均数来计算需要的领队数
        self.peds = []
        self.ped_to_group_dic = {}
        self.groups = []
        person_num = calculate_groups_person_num(self, person_num_sum)
        for i, num in enumerate(person_num):
            if not self.debug_mode and not self.random_init_mode:
                # 如果非debug_mode或者random_init_mode时，在固定点位创建行人
                self.peds.extend(self.factory.create_group_persons_in_radius(self.terrain, i, num, self.groups,
                                                                             self.ped_to_group_dic,
                                                                             self.group_size, self.debug_mode))
            elif not self.debug_mode and self.random_init_mode:
                self.peds.extend(
                    self.factory.random_create_persons(self.terrain, i, num, self.groups, self.ped_to_group_dic,
                                                       self.group_size, self.parser.start_point_dic, self.debug_mode))
            else:
                exit_type = self.terrain.get_random_exit(i)
                group = self.factory.create_people(self.terrain.start_points, exit_type, self.debug_mode)
                Group.set_group_process(group, self.groups, self.ped_to_group_dic, self.peds)
                self.peds.extend(group)
        self.left_person_num = sum(person_num)
        self.left_leader_num = self.agent_count
        self.not_arrived_peds = copy.copy(self.peds)

        self.elements = self.exits + self.obstacles + self.walls + self.not_arrived_peds
        # 得到一开始各个智能体距离出口的距离
        self.distance_to_exit.clear()
        self.get_peds_distance_to_exit()
        # 添加leader数组以供planner使用
        self.leaders = []
        idx = 0
        self.agents = copy.copy(self.possible_agents)
        self.agents_rev_dict = {}
        self.agents_dict = {}
        for ped in self.peds:
            if ped.is_leader:
                # 重置时针对pettingzoo
                self.agents_dict[self.agents[idx]] = ped
                self.agents_rev_dict[ped] = self.agents[idx]

                self.leaders.append(ped)
                idx += 1

    def setup_graphics(self):
        render_scale = 500, 500
        if self.viewer is not None:
            render_scale = self.viewer.get_render_scale()
        for ele in self.elements:
            ele.setup(self.batch, render_scale)

    def delete_person(self, per: Person):
        self.pop_from_render_list(per.id)
        self.left_person_num -= 1
        per.delete(self.world)
        if per.is_leader:
            self.left_leader_num -= 1

    def pop_from_render_list(self, person_id):
        """
        将行人从渲染队列中移除
        :param person_id:
        :return:
        """
        for idx, per in enumerate(self.elements):
            if per.type == ObjectType.Agent and per.id == person_id:
                self.elements.pop(idx)
                return
        raise Exception(u"移除一个不存在的行人!")

    def get_peds_distance_to_exit(self):
        for ped in self.peds:
            # min_dis = self.get_nearest_exit_dis((ped.getX, ped.getY))
            dis = self.get_ped_to_exit_dis((ped.getX, ped.getY), ped.exit_type)
            self.distance_to_exit.append(dis)
            self.points_in_last_step.append((ped.getX, ped.getY))

    @njit
    def get_ped_nearest_exit_dis(self, person_pos):
        x, y = person_pos
        min = math.inf
        for exit_point in self.terrain.exits:
            ex, ey = exit_point
            dis = sqrt(pow(ex - x, 2) + pow(ey - y, 2))
            min = dis if min > dis else min
        return min

    def get_ped_to_exit_dis(self, person_pos, exit_type):
        ex, ey = self.terrain.exits[exit_type - 3]  # 从3开始编号

        @njit
        def inner():
            nonlocal ex, ey
            x, y = person_pos
            return sqrt(pow(ex - x, 2) + pow(ey - y, 2))

        return inner()

    def get_ped_rel_pos_to_exit(self, person_pos, exit_type):
        ex, ey = self.terrain.exits[exit_type - 3]  # 从3开始编号
        x, y = person_pos
        return ex - x, ey - y

    def get_ped_nearest_elements(self, ped: Person, n: int, detect_range: float = 3.0,
                                 detect_type: ObjectType = ObjectType.Agent):
        detect_peds = ped.aabb_query(self.world, detect_range, detect_type)
        detect_peds = cast(List[Person], detect_peds)
        ret_elements = []
        mx, my = ped.getX, ped.getY
        for pe in detect_peds:
            ox, oy = pe.getX, pe.getY
            dis = ((ox - mx) ** 2 + (oy - my) ** 2) ** 0.5
            if dis <= detect_range and n > 0:
                ret_elements.append(pe)
                n -= 1
            if n <= 0:
                break
        return ret_elements

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Dict[
        str, Any]:
        self.seed(seed)
        self.reset_property()
        # 开始初始化环境
        self.initialize_env(self.terrain.map, self.terrain.map_spawn, person_num_sum=self.person_num)
        if self.person_handler.use_planner:
            self.person_handler.init_exit_kd_trees()  # 初始化KDTree以供后续使用
        # 添加初始观察状态
        init_obs = {}
        for ped in self.peds:
            if ped.is_leader:
                init_obs[self.agents_rev_dict[ped]] = (
                    self.person_handler.get_observation(ped, self.ped_to_group_dic[ped], 0))
        return init_obs

    def step(self, actions: Dict[str, int], planning_mode=False):
        is_done = {agent: False for agent in self.agents}
        truncated = {agentid: False for agentid in self.agents}
        # if len(actions) != self.agent_count: raise Exception("动作向量与智能体数量不匹配!")
        # if self.discrete:
        #     pass
        #     if len(actions[0]) != ACTION_DIM: raise Exception("动作向量的长度不正确!")
        # else:
        #     if len(actions[0]) != 2: raise Exception("动作向量的长度不正确!")
        # 清空上一步的碰撞状态

        for i in range(self.frame_skipping):
            # update box2d physical world
            for ped in self.not_arrived_peds:
                if ped.is_done and ped.has_removed:
                    continue
                belong_group = self.ped_to_group_dic[ped]
                if ped.is_leader:
                    # 是leader用强化学习算法来控制
                    self.person_handler.set_action(ped, actions[self.agents_rev_dict[ped]])
                else:
                    # 是follower用社会力模型来控制
                    self.person_handler.set_follower_action(ped,
                                                            actions[belong_group.id],
                                                            belong_group,
                                                            self.terrain.exits[ped.exit_type - 3])
                # 施加合力给行人
                ped.body.ApplyForceToCenter(b2Vec2(ped.total_force), wake=True)
                ped.total_force = np.zeros([2])
            self.world.Step(1 / TICKS_PER_SEC, vel_iters, pos_iters)
            self.world.ClearForces()
            for ped in self.peds:
                ped.update(self.exits, self.step_in_env, self.terrain.map)

            for group in self.groups:
                group.update()

        for ped in self.peds:
            if ped.is_done and not ped.has_removed:  # 移除到达出口的leader和follower
                logging.warning("Agent{}:Leave the exit{}!".format(ped.id, ped.exit_type))
                self.delete_person(ped)
                if ped.is_leader:
                    pass
                    # self.agents.remove(self.agents_rev_dict[ped]) # 为了tianshou框架的方便，这里将到达出口的人的is_done置为False，本来应该是True的！

        # 该环境中智能体是合作关系，因此使用统一奖励为好，此处使用了pettingzoo的形式
        obs, rewards = self.person_handler.step(self.peds, self.ped_to_group_dic, self.agents_rev_dict,
                                                int(self.step_in_env / self.frame_skipping))
        if self.disable_reward:
            for key in rewards.keys():
                rewards[key] = 0.0

        for idx, group in enumerate(self.groups):
            if group.leader.is_done:
                # is_done[self.agents_rev_dict[group.leader]] = True     # 为了tianshou框架的方便，这里将到达出口的人的is_done置为False，本来应该是True的！
                pass

        def is_done_operation():
            is_done = {agent: True for agent in self.agents}
            truncated = {agentid: False for agentid in self.agents}
            self.agents.clear()  # 为了tianshou框架的方便，这里将到达出口的人的is_done置为False，本来应该是True的！
            return is_done, truncated

        if planning_mode and self.left_leader_num < self.agent_count:
            # 在planning_mode下，一旦有leader到达出口就终止
            is_done, truncated = is_done_operation()

        if (not self.train_mode and self.left_person_num == 0) or (self.train_mode and self.left_leader_num == 0):
            is_done, truncated = is_done_operation()
            logging.warning(u"所有智能体到达出口!")

        self.step_in_env += self.frame_skipping
        if self.step_in_env > self.maxStep:  # 如果maxStep步都没有完全撤离，is_done直接为True
            is_done = {agent: True for agent in self.agents}
            truncated = {agentid: True for agentid in self.agents}
            # 清空agents里面的所有元素
            self.agents = []
            logging.warning(u"在{}步时强行重置环境!".format(self.maxStep))
        leader_pos = []
        leader_step = []
        for group in self.groups:
            leader_pos.append(group.leader.pos.tolist())  # 添加leader在最后一帧的位置
            leader_step.append(group.leader.exit_in_step)
        # info = {
        #     "leader_step": leader_step,
        #     "collision_with_wall": self.collide_wall,
        #     "collision_with_agents": self.collision_between_agents,
        #     "leader_pos": leader_pos
        # }

        # 为了有些MDP更新图像用
        self.person_handler.update_image_data()

        info = {agent: {} for i, agent in enumerate(self.possible_agents)}
        return obs, rewards, is_done, truncated, info

    def render(self, mode="human", ratio=1):
        if self.viewer is None:  # 如果调用了 render, 而且没有 viewer, 就生成一个
            self.viewer = PedsMoveEnvViewer(self, render_ratio=ratio)
        self.viewer.render()  # 使用 Viewer 中的 render 功能

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def observation_space(self, agent: str) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]

    def debug_step(self):
        if not self.once:
            for i, ped in enumerate(self.peds):
                print("Agent{}:".format(i))
                # ped.raycast(self.world, b2Vec2(1,0), test_mode=True)
                ped.aabb_query(self.world, 1, test_mode=True)
                print("############")
            self.once = True
