import copy
import random
import math
from collections import defaultdict
from typing import List

import pyglet
import math
import numpy as np

from Box2D import *
from math import sin, cos
from ped_env.utils.colors import ColorRed, exit_type_to_color, ColorYellow
from ped_env.functions import transfer_to_render, normalized, ij_power
from ped_env.utils.misc import FixtureInfo, ObjectType

class Agent():

    @property
    def getX(self):
        raise NotImplemented

    @property
    def getY(self):
        raise NotImplemented

    def self_driven_force(self, force):
        raise NotImplemented

class Person(Agent):
    body = None
    box = None

    radius = 0.4 / 2  # 设置所有行人直径为0.4米
    mass = 64 # 设置所有行人的质量为64kg
    alpha = 0.5
    A = 2000
    B = -0.08
    tau = 0.5
    Af = 0.01610612736
    Bf = 3.93216

    counter = 0  # 用于记录智能体编号
    pic = None

    def __init__(self,
                 env: b2World,
                 new_x,
                 new_y,
                 exit_type,
                 display_level,
                 debug_level,
                 desired_velocity = 2.4,
                 view_length = 5.0):
        '''

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
        '''
        super(Person, self).__init__()
        self.body = env.CreateDynamicBody(position=(new_x, new_y))
        self.exit_type = exit_type
        self.reward_in_episode = 0.0
        self.is_done = False
        self.has_removed = False

        # Add a fixture to it
        self.color = exit_type_to_color(self.exit_type)
        fixtureDef = b2FixtureDef()
        fixtureDef.shape = b2CircleShape(radius=self.radius)
        fixtureDef.density = self.mass / (math.pi * self.radius ** 2)
        fixtureDef.friction = 0.1 #指的是行人与墙以及其他行人间的摩擦
        fixtureDef.userData = FixtureInfo(Person.counter, self, ObjectType.Agent)
        self.id = Person.counter
        Person.counter += 1
        self.box = self.body.CreateFixture(fixtureDef)
        self.type = ObjectType.Agent
        self.view_length = view_length
        self.desired_velocity = desired_velocity

        self.display_level = display_level
        self.debug_level = debug_level

        self.collide_with_wall = False
        self.collide_with_agent = False

        self.is_leader = False

        #利用以空间换时间的方法，x,y每step更新一次
        self.x = self.body.position.x
        self.y = self.body.position.y
        self.pos = np.array([self.getX, self.getY])
        self.vec = np.array([0, 0])

        self.aabb_callback = AABBCallBack(self)
        self.raycast_callback = RaycastCallBack(self)

        self.total_force = np.zeros([2])

        # 通过射线得到8个方向上的其他行人与障碍物
        # 修复bug:未按照弧度值进行旋转
        identity = b2Vec2(1, 0)
        self.directions = []
        for angle in range(0, 360, int(360 / 8)):
            theta = np.radians(angle)
            mat = b2Mat22(cos(theta), -sin(theta),
                          sin(theta), cos(theta))
            vec = b2Mul(mat, identity)
            self.directions.append(vec)

    def update(self, exits):
        if self.is_done and self.has_removed:
            self.x, self.y = 0, 0
            self.pos = np.array([0, 0])
            self.vec = np.array([0, 0])
            return -1
        # 首先更新目前每个ped的坐标
        self.x, self.y = self.body.position.x, self.body.position.y
        self.pos = np.array([self.getX, self.getY])
        self.vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])
        c_agent, c_wall = False, False
        # 清空上一步的碰撞状态
        if self.collide_with_agent:
            c_agent = True
        if self.collide_with_wall:
            c_wall = True
        self.collide_with_wall = self.collide_with_agent = False

        # 检查是否有行人到达出口要进行移除
        def exam_self_exit(a, b):
            return b.exit_type == a.exit_type

        es = self.objects_query(exits, 1 + self.radius, exam_self_exit)
        if len(es) != 0:
            self.is_done = True
        return c_agent, c_wall

    def setup(self, batch, render_scale, test_mode=True):
        x, y = self.getX, self.getY
        if test_mode:
            # self.name_pic = pyglet.text.Label(str(self.id),
            #                                   font_name='Times New Roman',
            #                                   font_size=10,
            #                                   x=x * render_scale,y=y * render_scale,
            #                                   anchor_x='center', anchor_y='center',
            #                                   group=self.debug_level,
            #                                   batch=batch
            #                                   )
            pass
        self.pic = pyglet.shapes.Circle(x * render_scale, y * render_scale,
                                        self.radius * render_scale,
                                        color=self.color,
                                        batch=batch, group=self.display_level)
        if self.is_leader:
            self.leader_pic = pyglet.shapes.Circle(x * render_scale, y * render_scale,
                                        self.radius * 0.3 * render_scale,
                                        color=ColorYellow if self.color != ColorYellow else ColorRed,
                                        batch=batch, group=self.debug_level)

    def self_driven_force(self, direction):
        #给行人施加自驱动力，力的大小为force * self.desired_velocity * self.mass / self.tau
        d_v = direction * self.desired_velocity
        applied_force = (d_v - self.vec) * self.mass / self.tau
        self.total_force += applied_force
        #self.body.ApplyForceToCenter(applied_force, wake=True)

    def fij_force(self, peds, dic):
        def exam_if_self(a, b):
            return a.id != b.id
        detect_persons = self.objects_query(peds, 1 + self.radius * 2, exam_if_self)
        #detect_persons = self.aabb_query(world, 1 + self.radius * 2) # 如果两个行人距离超过1m，之间的作用力可以忽略不计
        if self.is_leader:
            for ped in detect_persons:
                if not ped.is_leader and dic[ped] == self:
                    continue
                else:
                    self.collide_with_agent = True
                    break
        total_force = b2Vec2(0, 0)
        for ped in detect_persons:
            pos, next_pos = (self.getX, self.getY), (ped.getX, ped.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            fij = self.A * math.exp((dis - self.radius - ped.radius)/self.B)
            total_force += b2Vec2(fij * (pos[0] - next_pos[0]), fij * (pos[1] - next_pos[1]))
        self.total_force += total_force
        #self.body.ApplyForceToCenter(total_force, wake=True)

    def fiw_force(self, obj):
        # detect_obstacles = self.aabb_query(world, 1 + self.radius * 2, detect_type=ObjectType.Obstacle)
        # detect_walls = self.aabb_query(world, 1 + self.radius * 2, detect_type=ObjectType.Wall)
        # detect_exits = self.aabb_query(world, 1 + self.radius * 2, detect_type=ObjectType.Exit)
        def exam_self_exit(a, b):
            if b.type != ObjectType.Exit:return True
            return b.exit_type != a.exit_type
        detect_things = self.objects_query(obj, 1 + self.radius * 2, exam_self_exit)
        #detect_obstacles = self.aabb_query(world, 1 + self.radius * 2, detect_type=ObjectType.Obstacle) # 如果行人与墙间的距离超过1m，之间的作用力可以忽略不计
        self.collide_with_wall = True if len(detect_things) != 0 else False
        total_force = b2Vec2(0, 0)
        for obs in detect_things:
            pos, next_pos = (self.getX, self.getY), (obs.getX, obs.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            fiw = self.A * math.exp((dis - self.radius - 0.5) / self.B) #因为每块墙的大小都为1*1m
            total_force += b2Vec2(fiw * (pos[0] - next_pos[0]), fiw * (pos[1] - next_pos[1]))
        self.total_force += total_force

    def ij_group_force(self, group):
        x = group.followers
        x.append(group.leader)
        total_ij_group_f = np.zeros([2])
        for target_ped in x:
            if target_ped == self:continue
            dis = group.dir_force_dic[self][target_ped][1]
            if dis < 0.37 or dis > 1.5:
                continue
            ij_group_f = group.dir_force_dic[self][target_ped][0]
            total_ij_group_f += ij_group_f
        self.total_force += total_ij_group_f

    def fraction_force(self):
        #给行人施加摩擦力，力的大小为-self.mass * velocity / self.tau
        vec = self.body.linearVelocity
        self.total_force += (-self.mass * vec / self.tau)

    SLOW_DOWN_DISTANCE = 0.6
    def arrive_force(self, target):
        now_point = np.array([self.getX, self.getY])
        target_point = np.array(target)
        now_vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])

        to_target = target_point - now_point
        distance = np.linalg.norm(to_target)
        if distance > self.SLOW_DOWN_DISTANCE:
            vec = normalized(to_target) * self.desired_velocity
            applied_force = vec - now_vec
        else:
            vec = to_target - now_vec
            applied_force = vec - now_vec
        applied_force = applied_force * self.desired_velocity * self.mass / self.tau
        self.total_force += applied_force
        #self.body.ApplyForceToCenter(applied_force, wake=True)

    def seek_force(self, target):
        now_point = np.array([self.getX, self.getY])
        target_point = np.array(target)
        now_vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])

        vec = np.linalg.norm(target_point - now_point)
        applied_force = vec - now_vec
        applied_force = applied_force * self.desired_velocity * self.mass / self.tau
        self.total_force += applied_force
        #self.body.ApplyForceToCenter(applied_force, wake=True)

    LEADER_BEHIND_DIST = 0.25
    def leader_follow_force(self, leader_body:b2Body):
        #计算目标点，并驱使arrive_force到达该点
        leader_vec = np.array([leader_body.linearVelocity.x, leader_body.linearVelocity.y])
        leader_pos = np.array([leader_body.position.x, leader_body.position.y])

        target = leader_pos + self.LEADER_BEHIND_DIST * normalized(-leader_vec)
        self.arrive_force(target)

    def evade_force(self, target_body:b2Body):
        now_point = np.array([self.getX, self.getY])
        vec = np.array([self.body.linearVelocity.x, self.body.linearVelocity.y])
        target_vec = np.array([target_body.linearVelocity.x, target_body.linearVelocity.y])
        target_point = np.array([target_body.position.x, target_body.position.y])
        to_target = target_point - now_point
        #计算向前预测的时间
        lookahead_time = np.linalg.norm(to_target) / (self.desired_velocity + np.linalg.norm(target_vec))
        #计算预期速度
        applied_force = normalized(now_point - (target_point + target_vec * lookahead_time)) - vec
        applied_force = applied_force * self.desired_velocity * self.mass / self.tau
        self.total_force += applied_force
        #self.body.ApplyForceToCenter(applied_force, wake=True)

    def evade_controller(self, leader:b2Body, evade_distance=0.5):
        '''
        :param leader:
        :param evade_distance_sqr: 躲避距离的平方值
        :return:
        '''
        #计算领队前方的一个点
        leader_pos = np.array([leader.position.x, leader.position.y])
        leader_vec = np.array([leader.linearVelocity.x, leader.linearVelocity.y])
        pos = np.array([self.getX, self.getY])
        leader_ahead = leader_pos + normalized(leader_vec) * self.LEADER_BEHIND_DIST
        #计算角色当前位置与领队前方某点的位置，如果小于某个值，就需要躲避
        dist = pos - leader_ahead
        if np.linalg.norm(dist) < evade_distance:
            self.evade_force(leader)

    leader_last_pos = None
    timer = 0
    def exam_leader_moved(self, leader:b2Body):
        moved = False
        if self.timer > 0:
            self.timer -= 1
            return moved
        if self.leader_last_pos is None:
            self.leader_last_pos = np.array([leader.position.x, leader.position.y])
            self.timer = 2
        else:
            now_pos = np.array([leader.position.x, leader.position.y])
            diff = 0.01
            if self.leader_last_pos[0] - diff < now_pos[0] < self.leader_last_pos[1] + diff \
                and self.leader_last_pos[1] - diff < now_pos[1] < self.leader_last_pos[1] + diff:
                self.timer = 2
            else:
                moved = True
            #if not is_done: self.leader_last_pos = now_pos
        return moved

    def aabb_query(self, world, size, detect_type:ObjectType = ObjectType.Agent, test_mode=False):
        '''
        进行以自己为中心，size大小的正方形区域检测
        :param world:
        :param size:
        :return:
        '''
        x, y = self.getX, self.getY

        callback = self.aabb_callback
        callback.radius = size
        callback.d_type = detect_type
        callback.detect_objects = []

        aabb = b2AABB()
        aabb.lowerBound = b2Vec2(x - size/2,y - size/2) #左上角坐标
        aabb.upperBound = b2Vec2(x + size/2,y + size/2) #右下角坐标
        world.QueryAABB(callback, aabb)
        return callback.detect_objects

    def objects_query(self, objects:List, size, conditionFunc = lambda self, obj:True):
        pos = (self.getX, self.getY)
        detect_objects = []
        for obj in objects:
            next_pos = (obj.getX, obj.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            if dis <= size and conditionFunc(self, obj):
                detect_objects.append(obj)
        return detect_objects

    def raycast(self, world:b2World, direction: b2Vec2, length = 5.0, test_mode=False):
        x, y = self.getX, self.getY
        start_point = b2Vec2(x, y)
        end_point = start_point + direction * length

        callback = self.raycast_callback
        callback.obs = length

        world.RayCast(callback, start_point, end_point)
        return callback.obs

    def delete(self, env:b2World):
        if self.pic != None:
            self.pic.delete()
            del (self.pic)
            if self.is_leader:
                self.leader_pic.delete()
                del (self.leader_pic)
        env.DestroyBody(self.body)
        self.has_removed = True
        del(self)

    def __str__(self):
        x, y = self.getX, self.getY
        return "id:{},x:{},y:{}".format(self.id, x, y)

    @property
    def getX(self):
        return self.x

    @property
    def getY(self):
        return self.y

class AABBCallBack(b2QueryCallback):
    def __init__(self, agent:Person):
        super(AABBCallBack, self).__init__()
        self.agent = agent
        self.d_type = None
        self.radius = None  # 只检测距离智能体中心为size的其他智能体和障碍物
        self.detect_objects = []

    def ReportFixture(self, fixture: b2Fixture):
        query_agent = (self.d_type == ObjectType.Agent and fixture.userData.id != self.agent.id and fixture.userData.type == ObjectType.Agent)
        query_obstacle = (self.d_type == ObjectType.Obstacle and fixture.userData.type == ObjectType.Obstacle)
        query_wall = (self.d_type == ObjectType.Wall and fixture.userData.type == ObjectType.Wall)
        query_exit = ((self.d_type == ObjectType.Exit and fixture.userData.type == ObjectType.Exit
                       and self.agent.exit_type != fixture.userData.env.exit_type)) #当出口不是自己的才有排斥力
        if query_agent or query_obstacle or query_wall or query_exit:  # 当
            pos = (self.agent.getX, self.agent.getY)
            next_pos = (fixture.userData.env.getX, fixture.userData.env.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            if dis <= self.radius:
                self.detect_objects.append(fixture.userData.env)
        return True

class RaycastCallBack(b2RayCastCallback):
    def __init__(self, agent:Person):
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

class BoxWall():
    body = None
    box = None
    height = 0.0
    width = 0.0

    counter = 0

    def __init__(self, env: b2World, new_x, new_y, new_width, new_height, display_level, object_type,
                 color=ColorRed):
        # new_x,new_y代表的是矩形墙的中心坐标
        self.body = env.CreateStaticBody(position=(new_x, new_y))
        self.x = self.body.position.x
        self.y = self.body.position.y
        self.width = new_width
        self.height = new_height
        self.color = color

        # And add a box fixture onto it
        self.box = self.body.CreatePolygonFixture(box=(new_width / 2, new_height / 2), density=0)
        self.box.userData = FixtureInfo(BoxWall.counter, self, object_type)
        BoxWall.counter += 1

        self.type = object_type
        self.display_level = display_level

    def setup(self, batch, render_scale):
        # pyglet以左下角那个点作为原点
        x, y, width, height = transfer_to_render(self.getX, self.getY, self.width, self.height,
                                                 render_scale)
        self.pic = pyglet.shapes.Rectangle(x, y, width, height, self.color, batch, group=self.display_level)

    def delete(self):
        del(self)

    @property
    def getX(self):
        return self.x

    @property
    def getY(self):
        return self.y

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

class Group():
    counter = 0
    group_force_magnitude_dic = defaultdict(float)
    def __init__(self, leader:Person, followers:List[Person]):
        self.id = Group.counter
        Group.counter += 1
        Group.get_gp_magnitude()
        self.leader = leader
        leader.is_leader = True
        self.followers = followers
        self.dir_force_dic = defaultdict(lambda : defaultdict(float))
        self._get_dir_force_between_groupers()

    @classmethod
    def get_gp_magnitude(cls):
        if len(Group.group_force_magnitude_dic) > 0:
            return
        for r in np.arange(0.37, 1.5, 0.01):
            Group.group_force_magnitude_dic[r] = ij_power(r)

    def is_done(self):
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

    def __get_distance(self, a, b):
        ax, ay = a.getX, a.getY
        bx, by = b.getX, b.getY
        return ((ax - bx)**2 + (ay - by)**2)**0.5

    def __get_group_center(self):
        center_x, center_y = self.leader.getX, self.leader.getY
        for ped in self.followers:
            center_x += ped.getX
            center_y += ped.getY
        center_x /= (len(self.followers) + 1)
        center_y /= (len(self.followers) + 1)
        return (center_x, center_y)

    def __get_nij(self, target, now):
        return normalized(target.pos - now.pos)

    def _get_dir_force_between_groupers(self):
        #先计算leader与各个follower的间距
        for follower in self.followers:
            dis = self.__get_distance(self.leader, follower)
            self.dir_force_dic[follower][self.leader] = [dis * self.__get_nij(self.leader, follower), dis]
        #按顺序计算各个follower的间距
        for i in range(len(self.followers)):
            fa = self.followers[i]
            for j in range(i + 1, len(self.followers)):
                fb = self.followers[j]
                dis = self.__get_distance(fa, fb)
                self.dir_force_dic[fa][fb] = [dis * self.__get_nij(fb, fa), dis]
                self.dir_force_dic[fb][fa] = [dis * self.__get_nij(fa, fb), dis]

    def update(self):
        #self._get_dir_force_between_groupers()
        self.__get_group_center()

    def get_group_force(self, pedA:Person, pedB:Person):
        key = (pedA, pedB)
        dis = round(self.dir_force_dic[key], 2)
        dis = max(min(dis, 0.37), 1.5)
        return Group.group_force_magnitude_dic[dis]








