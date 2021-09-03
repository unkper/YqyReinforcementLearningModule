import pyglet

from Box2D import *

from pyglet.graphics import Batch
from env.utils.colors import ColorRed
from env.functions import transfer_to_render
from env.utils.misc import FixtureInfo, ObjectType

SCALE = 10


class Agent():

    @property
    def getX(self):
        raise NotImplemented

    @property
    def getY(self):
        raise NotImplemented

    def move(self, force):
        raise NotImplemented


class Person(Agent):
    body = None
    box = None
    radius = 0.5 / 2  # 设置所有行人直径为0.5米

    counter = 0  # 用于记录智能体编号

    def __init__(self, env: b2World, new_x, new_y, color=ColorRed):
        super(Person, self).__init__()
        self.body = env.CreateDynamicBody(position=(new_x, new_y))
        self.color = color

        self.reward_in_episode = 0.0
        self.is_done = False

        self.has_removed = False

        # Add a fixture to it
        fixtureDef = b2FixtureDef()
        fixtureDef.shape = b2CircleShape(radius=self.radius)
        fixtureDef.density = 1
        fixtureDef.friction = 0.1
        fixtureDef.userData = FixtureInfo(Person.counter, self, ObjectType.Agent)
        self.id = Person.counter
        Person.counter += 1
        self.box = self.body.CreateFixture(fixtureDef)

        self.type = ObjectType.Agent

    def setup(self, batch: Batch):
        x, y = self.body.position.x, self.body.position.y
        self.pic = pyglet.shapes.Circle(x * SCALE, y * SCALE, self.radius * SCALE, color=self.color, batch=batch)

    def move(self, force):
        x, y = self.body.position.x, self.body.position.y
        self.body.ApplyForce(force, (x, y), wake=True)

    def moveTo(self, targetX, targetY):
        x, y = self.body.position.x, self.body.position.y
        force = b2Vec2(targetX - x, targetY - y)
        self.move(force)

    def delete(self):
        self.pic.delete()
        del (self.pic)
        self.has_removed = True

    def __str__(self):
        x, y = self.body.position.x, self.body.position.y
        return "x:{},y:{}".format(x, y)

    @property
    def getX(self):
        return self.body.position.x

    @property
    def getY(self):
        return self.body.position.y

    @classmethod
    def handle_collision(cls, objA, objB):
        pass


class BoxWall():
    body = None
    box = None
    height = 0.0
    width = 0.0

    counter = 0

    def __init__(self, env: b2World, new_x, new_y, new_width, new_height,
                 color=ColorRed):
        # new_x,new_y代表的是矩形墙的中心坐标
        self.body = env.CreateStaticBody(position=(new_x, new_y))
        self.width = new_width
        self.height = new_height
        self.color = color

        # And add a box fixture onto it
        self.box = self.body.CreatePolygonFixture(box=(new_width / 2, new_height / 2), density=0)
        self.box.userData = FixtureInfo(BoxWall.counter, self, ObjectType.Wall)
        BoxWall.counter += 1

        self.type = ObjectType.Wall

    def setup(self, batch: Batch):
        # pyglet以左下角那个点作为原点
        x, y, width, height = transfer_to_render(self.body.position.x, self.body.position.y, self.width, self.height,
                                                 SCALE)
        self.pic = pyglet.shapes.Rectangle(x, y, width, height, self.color, batch)

    def __repr__(self):
        x, y = self.body.position.x, self.body.position.y
        return "x:{},y:{},height:{},width:{}".format(x, y, self.height * SCALE, self.width * SCALE)


class Exit(BoxWall):

    def __init__(self, env: b2World, new_x, new_y, width, height, color=ColorRed):
        super(Exit, self).__init__(env, new_x, new_y, width, height, color)
        # And add a box fixture onto it
        fixtrueDef = b2FixtureDef()
        fixtrueDef.shape = b2PolygonShape(box=(width / 2, height / 2))
        fixtrueDef.density = 0
        fixtrueDef.isSensor = True
        fixtrueDef.userData = FixtureInfo(Exit.counter, self, ObjectType.Exit)
        Exit.counter += 1
        self.box = self.body.CreateFixture(fixtrueDef)

        self.type = ObjectType.Exit
