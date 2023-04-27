import copy
import random
import enum
import typing
from collections import defaultdict
from typing import List

import pyglet
import math
import numpy as np

from Box2D import *
from math import sin, cos

from numpy import ndarray

from ped_env.settings import DIRECTIONS, identity
from ped_env.utils.colors import ColorRed, exit_type_to_color, ColorYellow
from ped_env.functions import transfer_to_render, ij_power, normalize_vector, angle_between
from ped_env.utils.misc import FixtureInfo, ObjectType


class Agent:

    @property
    def getX(self):
        raise NotImplemented

    @property
    def getY(self):
        raise NotImplemented

    def self_driven_force(self, force):
        raise NotImplemented


class PersonState(enum.Enum):
    walk_to_goal = 1
    follow_leader = 2
    route_to_leader = 3
    route_to_exit = 4


class Person(Agent):
    body = None
    box = None

    radius = 0.4 / 2  # 设置所有行人直径为0.4米
    mass = 64  # 设置所有行人的质量为64kg
    alpha = 0.5  # alpha * control_dir + (1 - alpha) * leader_dir
    A = 2000
    B = -0.08
    tau = 0.5
    Af = 0.01610612736
    Bf = 3.93216

    counter = 0  # 用于记录智能体编号
    body_pic = None

    a_star_path = None  # 用于follow在找不到路时的A*策略使用

    MAX_SPEED = 1.8  # 指定最大的行人速度(m/s)

    def __init__(self,
                 env: b2World,
                 new_x,
                 new_y,
                 exit_type,
                 display_level,
                 debug_level,
                 desired_velocity=2.4,
                 view_length=5.0):
        """

        暂定观察空间为8个方向的射线传感器（只探测墙壁）与8个方向的射线传感器（只探测其他行人）与导航力的方向以及与终点的距离，类型为Box(-inf,inf,(18,))，
        动作空间当为离散空间时为类型为Discrete(5)代表不移动，向左，向右，向上，向下走，奖励的设置为碰到墙壁
        :param env:
        :param new_x: 生成位置x
        :param new_y: 生成位置y
        :param exit_type: 去往的出口编号
        :param display_level: pyglet显示的画面层
        :param debug_level: pyglet显示的画面层
        :param desired_velocity: 社会力模型中自驱动力相关的参数
        :param A: 社会力模型的参数A
        :param B: 社会力模型的参数B
        :param max_velocity:
        :param view_length: 智能体最远能观察到的距离
        :param tau: 社会力模型中关于地面摩擦和自驱动力的参数
        """
        super(Person, self).__init__()
        self.body = env.CreateDynamicBody(position=(new_x, new_y))
        self.body = typing.cast(b2BodyDef, self.body)
        self.body.allowSleep = True
        self.exit_type = exit_type
        self.reward_in_episode = 0.0
        self.is_done = False
        self.has_removed = False

        # Add a fixture to it
        self.color = exit_type_to_color(self.exit_type)
        fixtureDef = b2FixtureDef()
        fixtureDef.shape = b2CircleShape(radius=self.radius)
        fixtureDef.density = self.mass / (math.pi * self.radius ** 2)
        fixtureDef.friction = 0.1  # 指的是行人与墙以及其他行人间的摩擦
        fixtureDef.userData = FixtureInfo(Person.counter, self, ObjectType.Agent)
        self.id = Person.counter
        Person.counter += 1
        self.box = self.body.CreateFixture(fixtureDef)
        # 添加传感器用于社会力控制
        sensorDef = b2FixtureDef()
        sensorDef.shape = b2CircleShape(radius=self.radius + 1)  # 探测范围为1m
        sensorDef.isSensor = True
        sensorDef.userData = FixtureInfo(self.id, self, ObjectType.Sensor)
        self.sensor = self.body.CreateFixture(sensorDef)
        self.type = ObjectType.Agent
        self.view_length = view_length
        self.desired_velocity = desired_velocity

        self.display_level = display_level
        self.debug_level = debug_level

        self.collide_obstacles = {}
        self.collide_agents = {}
        # 此处指智能体的检测器和墙相撞
        self.detected_obstacles = {}
        self.detected_agents = {}

        self.is_leader = False

        # 利用以空间换时间的方法，x,y每step更新一次
        self.x = self.body.position.x
        self.y = self.body.position.y
        self.pos = np.array([self.getX, self.getY])
        self.vec = np.array([0, 0])

        self.aabb_callback = AABBCallBack(self)
        self.raycast_callback = RaycastCallBack(self)

        self.total_force = np.zeros([2])
        self.fij_force_last_eps = np.zeros([2])
        self.fiw_force_last_eps = np.zeros([2])

        self.person_state = PersonState.walk_to_goal
        self.group = None

        self.directions = DIRECTIONS

        self.exit_in_step = -1

    def update(self, exits, step_in_env, map: ndarray):
        if self.is_done and self.has_removed:
            self.x, self.y = 0, 0
            self.pos = np.array([0, 0])
            self.vec = np.array([0, 0])
            return -1
        # 首先更新目前每个ped的坐标
        self.x, self.y = self.body.position.x, self.body.position.y
        self.pos = np.array([self.getX, self.getY])
        self.vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])

        # 检查是否有行人到达出口要进行移除
        def exam_self_exit(a, b):
            return b.exit_type == a.exit_type

        out_of_edge = self.x < 0 or self.x >= map.shape[0] or self.y < 0 or self.y >= map.shape[1]
        es = self.objects_query(exits, 1 + self.radius, exam_self_exit)
        if len(es) != 0 or out_of_edge:
            self.is_done = True
            self.exit_in_step = step_in_env

    @property
    def vec_norm(self):
        return np.linalg.norm(self.vec)

    @property
    def vec_angle(self):
        if self.vec_norm == 0.0:
            return math.fmod(self.body.angle, math.pi * 2)
        else:
            return angle_between(identity, self.vec)

    def relative_distence(self, rpos):
        x, y = self.pos
        rx, ry = rpos
        return ((rx - x) ** 2 + (ry - y) ** 2) ** 0.5

    def relative_angle(self, rpos):
        x, y = self.pos
        rx, ry = rpos
        return angle_between(identity, np.array([rx - x, ry - y]))

    def setup(self, batch, render_scale, test_mode=True):
        x, y = self.getX, self.getY
        rx, ry = render_scale

        # print("Agent angle:{}".format(math.degrees(self.body.angle) % 360))
        if test_mode:
            pass
        self.body_pic = pyglet.shapes.Circle(x * rx, y * ry,
                                             self.radius * rx,
                                             color=self.color,
                                             batch=batch, group=self.display_level)

        # self.No_pic = pyglet.text.Label(str(self.id), )

        # t_len = self.radius * 1.5
        # cos_30 = -0.8660254
        # x1, y1 = x, (y + self.radius)
        # x2, y2 = x1 - t_len * 0.5, y1 + t_len * cos_30
        # x3, y3 = x1 + t_len * 0.5, y1 + t_len * cos_30
        # self.head = pyglet.shapes.Triangle(x1 * render_scale, y1 * render_scale, x2 * render_scale, y2 * render_scale,
        #                                    x3 * render_scale, y3 * render_scale, batch=batch, group=self.debug_level,
        #                                    color=self.color)
        if self.is_leader:
            self.leader_pic = pyglet.shapes.Star(x * rx, y * ry,
                                                 self.radius * 0.6 * rx,
                                                 self.radius * 0.4 * rx,
                                                 5,
                                                 color=ColorYellow if self.color != ColorYellow else ColorRed,
                                                 batch=batch, group=self.debug_level)

    def set_norm_velocity(self, action_type: int):
        from ped_env.settings import actions
        # 简单将速度设置为
        assert 0 <= action_type <= 8
        new_action = actions[action_type] * Person.MAX_SPEED
        self.body.linearVelocity = b2Vec2(new_action[0], new_action[1])

    def set_velocity(self, action_type: int):
        vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])
        c_speed_dictm = [0, -0.125, -0.25, -0.5, -1, 0.125, 0.25, 0.5, 1]
        c_speed_dict = [0, -0.125, -0.25, 0.125, 0.25, 0.5, 1]
        c_angle_dict = [(i * math.pi / 4) for i in c_speed_dictm]

        assert 0 <= action_type < 81
        old_velocity_norm = np.linalg.norm(vec)
        delta_velocity = c_speed_dictm[action_type // 9] * Person.MAX_SPEED / 2
        #delta_velocity = c_speed_dictm[action_type // 9] * old_velocity_norm
        delta_angle = c_angle_dict[action_type % 9]

        if old_velocity_norm == 0.0:
            old_angle_radian = self.body.angle  # 这里必须设置成当前朝向，不能为0.0否则出错！！！
        else:
            old_angle_radian = angle_between(identity, vec)

        if math.isnan(old_angle_radian):
            old_angle_radian = 0.0

        new_velocity_norm = max(min(old_velocity_norm + delta_velocity, Person.MAX_SPEED), 0)
        new_angle = old_angle_radian + delta_angle

        # print("vx:{}, vy:{}".format(new_velocity_norm * np.cos(new_angle),
        #                           new_velocity_norm * np.sin(new_angle)))

        self.body.linearVelocity = b2Vec2(new_velocity_norm * np.cos(new_angle), new_velocity_norm * np.sin(new_angle))

        # print("Agent {}:{}".format(self.id, self.body.linearVelocity))
        # print("new_v:{}, new_a:{}".format(new_velocity_norm, new_angle))

    def self_driven_force(self, direction):
        # 给行人施加自驱动力，力的大小为force * self.desired_velocity * self.mass / self.tau
        d_v = direction * self.desired_velocity
        applied_force = (d_v - self.vec) * self.mass / self.tau
        self.total_force += applied_force

    # 社会力模型添加
    def fij_force(self, peds, group):
        detect_persons = list(self.detected_agents.values())
        total_force = b2Vec2(0, 0)
        for ped in detect_persons:
            if self.is_leader and ped in group:
                continue
            pos, next_pos = (self.getX, self.getY), (ped.getX, ped.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            fij = self.A * math.exp((dis - self.radius - ped.radius) / self.B)
            if ped in group:
                fij *= 0.2
            total_force += b2Vec2(fij * (pos[0] - next_pos[0]), fij * (pos[1] - next_pos[1]))
        self.fij_force_last_eps = total_force
        self.total_force += total_force

    def fiw_force(self, obj):
        detect_things = list(self.detected_obstacles.values())
        total_force = b2Vec2(0, 0)
        for obs in detect_things:
            pos, next_pos = (self.getX, self.getY), (obs.getX, obs.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            fiw = self.A * math.exp((dis - self.radius - 0.5) / self.B)  # 因为每块墙的大小都为1*1m
            total_force += b2Vec2(fiw * (pos[0] - next_pos[0]), fiw * (pos[1] - next_pos[1]))
        self.fiw_force_last_eps = total_force
        self.total_force += total_force

    def ij_group_force(self, group):
        if self.is_leader:
            raise Exception("只能为follower添加成员力!")
        total_ij_group_f = group.get_group_force(self)
        self.total_force += total_ij_group_f

    def aabb_query(self, world, size, detect_type: ObjectType = ObjectType.Agent, test_mode=False):
        """
        进行以自己为中心，size大小的正方形区域检测
        :param world:
        :param size:
        :return:
        """
        x, y = self.getX, self.getY

        callback = self.aabb_callback
        callback.radius = size
        callback.d_type = detect_type
        callback.detect_objects = []

        aabb = b2AABB()
        aabb.lowerBound = b2Vec2(x - size / 2, y - size / 2)  # 左上角坐标
        aabb.upperBound = b2Vec2(x + size / 2, y + size / 2)  # 右下角坐标
        world.QueryAABB(callback, aabb)
        return callback.detect_objects

    def objects_query(self, objects: List, size, conditionFunc=lambda self, obj: True):
        pos = (self.getX, self.getY)
        detect_objects = []
        for obj in objects:
            next_pos = (obj.getX, obj.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            if dis <= size and conditionFunc(self, obj):
                detect_objects.append(obj)
        return detect_objects

    def raycast(self, world: b2World, direction: b2Vec2, length=5.0, test_mode=False):
        x, y = self.getX, self.getY
        start_point = b2Vec2(x, y)
        end_point = start_point + direction * length

        callback = self.raycast_callback
        callback.obs = length

        world.RayCast(callback, start_point, end_point)
        return callback.obs

    def delete(self, env: b2World):
        if self.body_pic != None:
            self.body_pic.delete()
            del self.body_pic
            if self.is_leader:
                self.leader_pic.delete()
                del self.leader_pic
        if self.body != None:
            env.DestroyBody(self.body)
        # self.has_removed = True  # 因为逻辑可能不对，选择了在奖励模型处设置has_removed

    def __str__(self):
        x, y = self.getX, self.getY
        return "id:{},x:{},y:{}".format(self.id, x, y)

    def __repr__(self):
        return "Agent{}".format(self.id)

    @property
    def getX(self):
        return self.x

    @property
    def getY(self):
        return self.y


class AABBCallBack(b2QueryCallback):
    def __init__(self, agent: Person):
        super(AABBCallBack, self).__init__()
        self.agent = agent
        self.d_type = None
        self.radius = None  # 只检测距离智能体中心为size的其他智能体和障碍物
        self.detect_objects = []

    def ReportFixture(self, fixture: b2Fixture):
        query_agent = (
                self.d_type == ObjectType.Agent and fixture.userData.id != self.agent.id and fixture.userData.type == ObjectType.Agent)
        query_obstacle = (self.d_type == ObjectType.Obstacle and fixture.userData.type == ObjectType.Obstacle)
        query_wall = (self.d_type == ObjectType.Wall and fixture.userData.type == ObjectType.Wall)
        query_exit = ((self.d_type == ObjectType.Exit and fixture.userData.type == ObjectType.Exit
                       and self.agent.exit_type != fixture.userData.model.exit_type))  # 当出口不是自己的才有排斥力
        if query_agent or query_obstacle or query_wall or query_exit:  # 当
            pos = (self.agent.getX, self.agent.getY)
            next_pos = (fixture.userData.model.getX, fixture.userData.model.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            if dis <= self.radius:
                self.detect_objects.append(fixture.userData.model)
        return True


class RaycastCallBack(b2RayCastCallback):
    def __init__(self, agent: Person):
        super().__init__()
        self.obs = None
        self.agent = agent

    def ReportFixture(self, fixture: b2Fixture, point, normal, fraction) -> float:
        if fixture.userData.type == ObjectType.Exit:  # 不对类型为Exit的物体进行检测
            return fraction
        pos = (self.agent.getX, self.agent.getY)
        obs = ((point[0] - pos[0]) ** 2 + (point[1] - pos[1]) ** 2) ** 0.5
        self.obs = min(5.0, self.obs, obs)
        return fraction


class BoxWall:
    body = None
    box = None
    height = 0.0
    width = 0.0

    BOX_WALL_HEIGHT = BOX_WALL_WIDTH = 1
    PIECE_WALL_HEIGHT = 1
    PIECE_WALL_WIDTH = 0.2

    counter = 0

    def __init__(self, env: b2World, new_x, new_y, new_width, new_height, display_level, object_type,
                 color=ColorRed):
        # new_x,new_y代表的是矩形墙的中心坐标
        self.body: b2BodyDef = env.CreateStaticBody(position=(new_x, new_y))
        self.body.allowSleep = True
        self.x = self.body.position.x
        self.y = self.body.position.y
        self.width = new_width
        self.height = new_height
        self.color = color

        # And add a box fixture onto it
        self.box = self.body.CreatePolygonFixture(box=(new_width / 2, new_height / 2), density=0)
        self.box.userData = FixtureInfo(BoxWall.counter, self, object_type)
        self.id = BoxWall.counter
        BoxWall.counter += 1

        self.type = object_type
        self.display_level = display_level

    def setup(self, batch, render_scale):
        # pyglet以左下角那个点作为原点
        x, y, width, height = transfer_to_render(self.getX, self.getY, self.width, self.height,
                                                 render_scale)
        self.pic = pyglet.shapes.Rectangle(x, y, width, height, self.color, batch, group=self.display_level)

    def delete(self):
        del self

    @property
    def getX(self):
        return self.x

    @property
    def getY(self):
        return self.y

    def __repr__(self):
        return "BoxWall{}".format(self.id)


class Exit(BoxWall):
    counter = 0

    def __init__(self, env: b2World, new_x, new_y, exit_type, width, height, display_level):
        color = exit_type_to_color(exit_type)
        super(Exit, self).__init__(env, new_x, new_y, width, height, display_level, ObjectType.Exit, color)
        # And add a box fixture onto it
        fixtrueDef = b2FixtureDef()
        fixtrueDef.shape = b2PolygonShape(box=(width / 2, height / 2))
        fixtrueDef.density = 0
        fixtrueDef.isSensor = True
        fixtrueDef.userData = FixtureInfo(Exit.counter, self, ObjectType.Exit)
        Exit.counter += 1
        self.box = self.body.CreateFixture(fixtrueDef)

        self.exit_type = exit_type

    def __repr__(self):
        return "Exit{}".format(self.id)


class Group:
    counter = 0
    group_force_magnitude_dic = defaultdict(float)

    # 通过设置这个值来决定智能体跟随leader的点位
    LEADER_BEHIND_DIST = 0.25

    def __init__(self, leader: Person, followers: List[Person]):
        self.id = Group.counter
        Group.counter += 1
        Group.get_gp_magnitude()
        self.leader = leader
        leader.is_leader = True
        self.followers = followers
        self.followers_set = set(followers)
        self.members = followers.copy()
        self.members.append(leader)
        self.members_set = set(self.members)
        self.dir_force_dic = defaultdict(lambda: defaultdict(float))
        self.group_center = self.__get_group_center()
        self._get_group_force_dis_nij()

    @classmethod
    def get_gp_magnitude(cls):
        if len(Group.group_force_magnitude_dic) > 0:
            return
        for r in np.arange(0.37, 1.5, 0.01):
            Group.group_force_magnitude_dic[r] = ij_power(r)

    def is_done(self):
        """
        判断智能体是否到达终点
        """
        is_done = True and self.leader.is_done
        for ped in self.followers:
            is_done = is_done and ped.is_done
        return is_done

    def change_leader(self):
        last_leader = self.leader
        new_leader = random.sample(self.followers, 1)[0]
        self.leader = new_leader
        self.followers.remove(new_leader)
        new_leader.is_leader = True
        self.followers.append(last_leader)
        last_leader.is_leader = False

    @classmethod
    def __get_distance(cls, a, b):
        ax, ay = a.getX, a.getY
        bx, by = b[0], b[1]
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def __get_group_center(self):
        center_x, center_y = self.leader.getX, self.leader.getY
        for ped in self.followers:
            center_x += ped.getX
            center_y += ped.getY
        center_x /= (len(self.followers) + 1)
        center_y /= (len(self.followers) + 1)
        return (center_x, center_y)

    def __get_nij(self, target, now):
        return normalize_vector(target - now.pos)

    def get_distance_to_leader(self, ped: Person):
        lx, ly = self.leader.getX, self.leader.getY
        gx, gy = ped.getX, ped.getY
        return ((lx - gx) ** 2 + (ly - gy) ** 2) ** 0.5

    def _get_group_force_dis_nij(self):
        # 先计算leader身后一定间距的点与各个follower的间距
        lv, lpos = self.leader.vec, self.leader.pos
        fpos = lpos - lv * Group.LEADER_BEHIND_DIST
        for follower in self.followers:
            dis = self.__get_distance(follower, fpos)
            self.dir_force_dic[follower][self.leader] = [dis * self.__get_nij(fpos, follower), dis]

    def update(self):
        self.group_center = self.__get_group_center()
        self._get_group_force_dis_nij()

    def get_group_force(self, follower: Person):
        if follower not in self.followers_set:
            raise Exception("跟随者")
        nij, dis = self.dir_force_dic[follower][self.leader]
        dis = np.clip(np.round(dis, 2), 0.37, 1.5)
        return nij * Group.group_force_magnitude_dic[dis]

    def __contains__(self, item):
        return item in self.members_set

    def __repr__(self):
        return "Group{}".format(self.id)

    @classmethod
    def set_group_process(cls, group, groups, group_dic, persons):
        leader = random.sample(group, 1)[0]  # 随机选取一人作为leader
        followers = copy.copy(group)
        followers.remove(leader)
        group_obj = Group(leader, followers)
        for member in group:
            member.group = group_obj
        groups.append(group_obj)
        group_dic.update({per: group_obj for per in group})  # 将leader和follow的映射关系添加
        persons.extend(group)

    # def setup(self, batch, render_scale):
    #     self.pic = pyglet.shapes.Circle(self.group_center[0], self.group_center[1], 0.5, color=(0, 0, 255), batch=batch)
