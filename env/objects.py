import pyglet


from Box2D import *
from math import sin, cos
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
        '''
        :param env:
        :param new_x:
        :param new_y:
        :param color:
        暂定观察空间为8个方向的射线传感器（只探测墙壁）与8个方向的射线传感器（只探测其他行人）与导航力的方向以及与终点的距离，类型为Box(-inf,inf,(18,))，
        动作空间当为离散空间时为类型为Discrete(5)代表不移动，向左，向右，向上，向下走，奖励的设置为碰到墙壁
        '''
        super(Person, self).__init__()
        self.body = env.CreateDynamicBody(position=(new_x, new_y))
        self.color = color

        self.observation = []
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

    def get_observation(self, world:b2World):
        #通过射线得到8个方向上的其他行人与障碍物
        identity = b2Vec2(1, 0)
        directions = []
        for angle in range(0, 360, int(360/8)):
            mat = b2Mat22([cos(angle), sin(angle),
                           -sin(angle), cos(angle)])
            directions.append(b2Mul(mat, directions))
        #依次得到8个方向上的障碍物,在回调函数中体现，每次调用该函数都会给observation数组中添加两个值，分别代表该方向上最近的障碍物有多远（5米代表不存在）
        for i in range(8):
            self.raycast(world, directions[i])




    def move(self, force):
        x, y = self.body.position.x, self.body.position.y
        self.body.ApplyForce(force, (x, y), wake=True)

    def moveTo(self, targetX, targetY):
        x, y = self.body.position.x, self.body.position.y
        force = b2Vec2(targetX - x, targetY - y)
        self.move(force)

    def raycast(self, world:b2World, direction: b2Vec2, length = 5):
        x, y = self.getX, self.getY
        start_point = b2Vec2(x, y)
        end_point = start_point + direction * length

        class CallBack(b2RayCastCallback):
            def __init__(self, obs):
                self.obs = obs

            def ReportFixture(self, fixture: b2Fixture, point, normal, fraction) -> float:
                print(fixture.userData)
                return 1

        callback = CallBack(self.observation)

        world.RayCast(callback, start_point, end_point)
        return start_point, end_point

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
    counter = 0

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
