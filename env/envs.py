import copy
import random
import gym
import pyglet
import numpy
import Box2D as b2d

from Box2D import (b2World)
from typing import List, Tuple
from pyglet.window import key

from env.objects import BoxWall, Person, Exit
from env.utils.colors import (ColorBlue, ColorWall, ColorRed)
from env.event_listeners import MyContactListener
from env.utils.misc import ObjectType
from env.utils.maps import map1

TICKS_PER_SEC = 60
vel_iters, pos_iters = 6, 2


class Model():
    def __init__(self):
        self.world = None

    def find_person(self, person_id) -> Person:
        raise NotImplemented

    def pop_person_from_renderlist(self, person_id):
        raise NotImplemented


class PedsMoveEnv(Model, gym.Env):
    def __init__(self, window, render: bool = True):
        super(PedsMoveEnv, self).__init__()
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.listener = MyContactListener(self)
        self.world.contactListener = self.listener
        self.window = window
        self.batch = pyglet.graphics.Batch()
        self.render = render

    def start(self, maps:numpy.ndarray, person_num:List=[30,30], person_create_radius:float = 5):
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
        self.obstacles = self.create_walls(start_nodes_obs, (1, 1), Color=ColorBlue)
        self.exits = self.create_walls(start_nodes_exit, (1, 1), Color=ColorRed, CreateClass=Exit)# 创建出口
        self.walls = self.create_walls(start_nodes_wall, (1, 1))# 建造围墙
        # 随机初始化行人点
        self.peds = self.random_create_persons((5, 20), person_create_radius, person_num[0]) \
                    + self.random_create_persons((5, 30), person_create_radius, person_num[1])
        self.render_peds = copy.copy(self.peds)
        self.elements = self.exits + self.obstacles + self.walls + self.render_peds

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

    def setup_graphics(self):
        for ele in self.elements:
            ele.setup(self.batch)

    def find_person(self, person_id) -> Person:
        for per in self.peds:
            if per.id == person_id:
                return per
        return None

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

    def delete_person(self, per: Person):
        self.world.DestroyBody(per.body)
        self.pop_person_from_renderlist(per.id)
        per.delete()

    def update(self):
        # update box2d physical world
        self.world.Step(1 / TICKS_PER_SEC, vel_iters, pos_iters)
        self.world.ClearForces()

        # Just for test
        for i, ped in enumerate(self.peds):
            ped.move((random.random() * 1 - 0.5, random.random() * 1 - 0.5))
            # ped.moveTo(5, 25)
        # 检查是否有行人到达出口要进行移除
        for per in self.peds:
            if per.is_done and not per.has_removed:
                self.delete_person(per)

        if self.render:
            self.setup_graphics()

    #Gym Functions
    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass

class Box2DEnv1(pyglet.window.Window):
    def __init__(self):
        super().__init__(config=pyglet.gl.Config(double_buffer=True))

        self.model = PedsMoveEnv(self)
        self.width = 500
        self.height = 500

        # This call schedules the `update()` method to be called
        # TICKS_PER_SEC. This is the main game event loop.
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SEC)

    def start(self):
        self.model.start(map1)

    def setup(self):
        pyglet.graphics.glClearColor(255, 255, 255, 0)

    def update(self, dt):
        self.model.update()

    def on_draw(self):
        self.clear()
        self.model.batch.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.UP:
            self.model.peds[0].raycast(self.model.world,b2d.b2Vec2(0, 1))
        elif symbol == key.DOWN:
            self.model.peds[0].raycast(self.model.world,b2d.b2Vec2(0, -1))
        elif symbol == key.LEFT:
            self.model.peds[0].raycast(self.model.world,b2d.b2Vec2(1, 0))
        elif symbol == key.RIGHT:
            self.model.peds[0].raycast(self.model.world,b2d.b2Vec2(-1, 0))
