import pyglet
import math

from Box2D import *
from math import sin, cos
from ped_env.utils.colors import ColorRed
from ped_env.functions import transfer_to_render
from ped_env.utils.misc import FixtureInfo, ObjectType

SCALE = 10


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
    radius = 0.5 / 2  # 设置所有行人直径为0.5米
    mass = 80 # 设置所有行人的质量为80kg

    counter = 0  # 用于记录智能体编号
    pic = None

    def __init__(self,
                 env: b2World,
                 new_x,
                 new_y,
                 color=ColorRed,
                 desired_velocity = 1.0,
                 max_velocity = 1.6,
                 view_length = 5.0):
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
        self.reward_in_episode = 0.0
        self.is_done = False
        self.has_removed = False

        # Add a fixture to it
        self.color = color
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
        self.max_velocity = max_velocity
        self.damping = 0.2

        self.collide_with_wall = False
        self.collide_with_agent = False

        # 通过射线得到8个方向上的其他行人与障碍物
        identity = b2Vec2(1, 0)
        self.directions = []
        for angle in range(0, 360, int(360 / 8)):
            mat = b2Mat22(cos(angle), -sin(angle),
                          sin(angle), cos(angle))
            vec = b2Mul(mat, identity)
            self.directions.append(vec)

    def clamp_velocity(self):
        vec = self.body.linearVelocity
        if vec.length > self.max_velocity:
            vec.Normalize()
            self.body.linearVelocity = vec * self.max_velocity

    def setup(self, batch, extra_scale=1.0):
        x, y = self.body.position.x, self.body.position.y
        self.pic = pyglet.shapes.Circle(x * SCALE, y * SCALE, self.radius * SCALE * extra_scale, color=self.color, batch=batch)

    def get_observation(self, world:b2World):
        observation = []
        if self.is_done:
            #根据论文中方法，给予一个零向量加智能体的id
            observation.append(self.id)
            observation.extend([0.0 for _ in range(8)]) #5个方向上此时都不应该有障碍物
            observation.extend([0.0, 0.0]) #将智能体在出口的位置赋予
            observation.extend([0.0, 0.0]) #智能体的速度设置为0
            return observation
        observation.append(self.id)

        #依次得到8个方向上的障碍物,在回调函数中体现，每次调用该函数都会给observation数组中添加值，分别代表该方向上最近的障碍物有多远（5米代表不存在）
        for i in range(8):
            temp = self.raycast(world, self.directions[i], self.view_length)
            observation.append(temp)
        #给予智能体当前位置
        observation.append(self.getX)
        observation.append(self.getY)
        #给予智能体当前速度
        vec = self.body.linearVelocity
        observation.append(vec.x)
        observation.append(vec.y)
        return observation

    def self_driven_force(self, force):
        #给行人施加自驱动力，力的大小为force * self.desired_velocity * self.mass / self.toi
        x, y = self.body.position.x, self.body.position.y
        applied_force = force * self.desired_velocity * self.mass
        self.body.ApplyLinearImpulse(applied_force, (x, y), wake=True)

    def fraction_force(self):
        #给行人施加摩擦力，力的大小为-self.mass * velocity / self.toi
        x, y = self.body.position.x, self.body.position.y
        vec = self.body.linearVelocity
        self.body.ApplyLinearImpulse(-self.mass * vec * self.damping, (x, y), wake=True)

    def raycast(self, world:b2World, direction: b2Vec2, length = 5.0):
        x, y = self.getX, self.getY
        start_point = b2Vec2(x, y)
        end_point = start_point + direction * length
        class CallBack(b2RayCastCallback):
            def __init__(self, pos):
                super().__init__()
                self.obs = 5.0
                self.pos = pos

            def ReportFixture(self, fixture: b2Fixture, point, normal, fraction) -> float:
                self.obs = ((point[0] - self.pos[0]) ** 2 + (point[1] - self.pos[1]) ** 2) ** 0.5
                self.obs = min(5.0, self.obs)
                return fraction
        callback = CallBack((x, y))

        world.RayCast(callback, start_point, end_point)
        return callback.obs

    def delete(self):
        if self.pic != None:
            self.pic.delete()
            del (self.pic)
        self.has_removed = True
        del(self)

    def __str__(self):
        x, y = self.body.position.x, self.body.position.y
        return "x:{},y:{}".format(x, y)

    @property
    def getX(self):
        return self.body.position.x

    @property
    def getY(self):
        return self.body.position.y

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

    def setup(self, batch, extra_scale = 1.0):
        # pyglet以左下角那个点作为原点
        x, y, width, height = transfer_to_render(self.body.position.x, self.body.position.y, self.width, self.height,
                                                 SCALE * extra_scale)
        self.pic = pyglet.shapes.Rectangle(x, y, width, height, self.color, batch)

    def delete(self):
        del(self)

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
