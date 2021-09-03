import copy
import random
import sys
import pyglet
import Box2D as b2d

from Box2D import (b2World)
from typing import List, Tuple
from pyglet.window import key

from env.objects import BoxWall, Person, Exit
from env.utils.colors import (ColorBlue, ColorWall, ColorRed)
from env.event_listeners import MyContactListener
from env.utils.misc import ObjectType

TICKS_PER_SEC = 60
vel_iters, pos_iters = 6, 2


class Model():
    def __init__(self):
        self.world = None

    def find_person(self, person_id) -> Person:
        raise NotImplemented

    def pop_person_from_renderlist(self, person_id):
        raise NotImplemented


class Box2DEnv1_Model(Model):
    def __init__(self, window, render: bool = True):
        super(Box2DEnv1_Model, self).__init__()
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.listener = MyContactListener(self)
        self.world.contactListener = self.listener
        self.window = window
        self.batch = pyglet.graphics.Batch()
        self.render = render

    def start(self):
        self.obstacle = BoxWall(self.world, 12.5, 25, 5, 30, ColorBlue)
        start_nodes = [(0, 25), (50, 25), (25, 50), (25, 0)]
        width_height = [(1, 50), (1, 50), (50, 1), (50, 1)]
        self.walls = [BoxWall(self.world, start_nodes[i][0],
                              start_nodes[i][1], width_height[i][0],
                              width_height[i][1], ColorWall) for i in range(len(start_nodes))]
        # 随机初始化行人点
        start_nodes = [(5, 15), (4, 15), (5, 35), (4, 35), (3, 35)]
        self.peds = [Person(self.world, start_nodes[i][0],
                            start_nodes[i][1]) for i in range(len(start_nodes))]
        self.render_peds = copy.copy(self.peds)
        start_nodes = [(50, 35), (50, 15)]
        width_height = [(2, 10), (2, 10)]
        self.exits = [Exit(self.world, start_nodes[i][0],
                           start_nodes[i][1], width_height[i][0],
                           width_height[i][1], ColorRed) for i in range(len(start_nodes))]

        self.elements = self.exits + [self.obstacle] + self.walls + self.render_peds

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

    def raycast(self, person_id: int, direction: b2d.b2Vec2) -> Tuple[b2d.b2Vec2, b2d.b2Vec2]:
        person = self.find_person(person_id)
        x, y = person.getX, person.getY
        start_point = b2d.b2Vec2(x, y)
        end_point = start_point + direction * 100

        class CallBack(b2d.b2RayCastCallback):
            def ReportFixture(self, fixture: b2d.b2Fixture, point, normal, fraction)->float:
                print(fixture.userData)
                return 1
        callback = CallBack()

        self.world.RayCast(callback, start_point, end_point)
        return start_point, end_point

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


class Box2DEnv1(pyglet.window.Window):
    def __init__(self):
        super().__init__(config=pyglet.gl.Config(double_buffer=True))

        self.model = Box2DEnv1_Model(self)
        self.width = 500
        self.height = 500

        # This call schedules the `update()` method to be called
        # TICKS_PER_SEC. This is the main game event loop.
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SEC)

    def start(self):
        self.model.start()

    def setup(self):
        pyglet.graphics.glClearColor(255, 255, 255, 0)

    def update(self, dt):
        self.model.update()

    def on_draw(self):
        self.clear()
        self.model.batch.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.UP:
            self.model.raycast(0, b2d.b2Vec2(0, 1))
        elif symbol == key.DOWN:
            self.model.raycast(0, b2d.b2Vec2(0, -1))
        elif symbol == key.LEFT:
            self.model.raycast(0, b2d.b2Vec2(-1, 0))
        elif symbol == key.RIGHT:
            self.model.raycast(0, b2d.b2Vec2(1, 0))
