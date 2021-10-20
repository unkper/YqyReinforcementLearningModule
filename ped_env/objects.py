import pyglet
import math

from Box2D import *
from math import sin, cos
from ped_env.utils.colors import ColorRed, exit_type_to_color
from ped_env.functions import transfer_to_render
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
    radius = 0.5 / 2  # 设置所有行人直径为0.5米
    mass = 80 # 设置所有行人的质量为80kg

    counter = 0  # 用于记录智能体编号
    pic = None

    def __init__(self,
                 env: b2World,
                 new_x,
                 new_y,
                 exit_type,
                 display_level,
                 debug_level,
                 desired_velocity = 10.0,
                 A = 30,
                 B = -0.08,
                 max_velocity = 1.6,
                 view_length = 5.0,
                 tau = 0.5):
        '''
        :param env:
        :param new_x:
        :param new_y:
        :param color:
        社会力模型的两个参数A,B
        暂定观察空间为8个方向的射线传感器（只探测墙壁）与8个方向的射线传感器（只探测其他行人）与导航力的方向以及与终点的距离，类型为Box(-inf,inf,(18,))，
        动作空间当为离散空间时为类型为Discrete(5)代表不移动，向左，向右，向上，向下走，奖励的设置为碰到墙壁
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
        self.A = A
        self.B = B
        self.max_velocity = max_velocity
        self.tau = tau

        self.display_level = display_level
        self.debug_level = debug_level

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

    def setup(self, batch, render_scale, test_mode=True):
        x, y = self.body.position.x, self.body.position.y
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

    def self_driven_force(self, force):
        #给行人施加自驱动力，力的大小为force * self.desired_velocity * self.mass / self.tau
        x, y = self.body.position.x, self.body.position.y
        applied_force = force * self.desired_velocity * self.mass / self.tau
        self.body.ApplyForce(applied_force, (x, y), wake=True)
        #self.body.ApplyLinearImpulse(applied_force, (x, y), wake=True)

    def fraction_force(self):
        #给行人施加摩擦力，力的大小为-self.mass * velocity / self.tau
        x, y = self.body.position.x, self.body.position.y
        vec = self.body.linearVelocity
        self.body.ApplyForce(-self.mass * vec / self.tau, (x, y), wake=True)
        #self.body.ApplyLinearImpulse(-self.mass * vec * self.damping, (x, y), wake=True)

    def fij_force(self, world):
        detect_persons = self.aabb_query(world, 1 + self.radius * 2) # 如果两个行人距离超过1m，之间的作用力可以忽略不计
        total_force = b2Vec2(0, 0)
        for ped in detect_persons:
            pos, next_pos = (self.getX, self.getY), (ped.getX, ped.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            fij = self.A * math.exp((dis - self.radius - ped.radius)/self.B)
            total_force += b2Vec2(fij * (pos[0] - next_pos[0]), fij * (pos[1] - next_pos[1]))
        x, y = self.body.position.x, self.body.position.y
        self.body.ApplyForce(total_force, (x, y), wake=True)

    def fiw_force(self, world):
        detect_obstacles = self.aabb_query(world, 1 + self.radius * 2, detect_type=ObjectType.Obstacle) # 如果行人与墙间的距离超过1m，之间的作用力可以忽略不计
        total_force = b2Vec2(0, 0)
        for obs in detect_obstacles:
            pos, next_pos = (self.getX, self.getY), (obs.getX, obs.getY)
            dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
            fiw = self.A * math.exp((dis - self.radius - 0.5) / self.B) #因为每块墙的大小都为1*1m
            total_force += b2Vec2(fiw * (pos[0] - next_pos[0]), fiw * (pos[1] - next_pos[1]))
        x, y = self.body.position.x, self.body.position.y
        self.body.ApplyForce(total_force, (x, y), wake=True)

    def aabb_query(self, world, size, detect_type:ObjectType = ObjectType.Agent, test_mode=False):
        '''
        进行以自己为中心，size大小的正方形区域检测
        :param world:
        :param size:
        :return:
        '''
        x, y = self.getX, self.getY
        class CallBack(b2QueryCallback):
            def __init__(self, pos, agent_id, radius, d_type):
                super(CallBack, self).__init__()
                self.pos = pos
                self.id = agent_id
                self.d_type = d_type
                self.radius = radius #只检测距离智能体中心为size的其他智能体和障碍物
                self.detect_objects = []

            def ReportFixture(self, fixture: b2Fixture):
                if test_mode:
                    print(fixture.userData.type)
                if (self.d_type == ObjectType.Agent and fixture.userData.id != self.id and fixture.userData.type == ObjectType.Agent) or \
                    (self.d_type == ObjectType.Obstacle and fixture.userData.type == ObjectType.Obstacle): #当
                    pos = self.pos
                    next_pos = (fixture.userData.model.getX, fixture.userData.model.getY)
                    dis = ((pos[0] - next_pos[0]) ** 2 + (pos[1] - next_pos[1]) ** 2) ** 0.5
                    if dis <= self.radius:
                        self.detect_objects.append(fixture.userData.model)
                return True
        callback = CallBack((x, y), self.id, size, detect_type)

        aabb = b2AABB()
        aabb.lowerBound = b2Vec2(x - size/2,y - size/2) #左上角坐标
        aabb.upperBound = b2Vec2(x + size/2,y + size/2) #右下角坐标
        world.QueryAABB(callback, aabb)
        return callback.detect_objects

    def raycast(self, world:b2World, direction: b2Vec2, length = 5.0, test_mode=False):
        x, y = self.getX, self.getY
        start_point = b2Vec2(x, y)
        end_point = start_point + direction * length
        class CallBack(b2RayCastCallback):
            def __init__(self, pos):
                super().__init__()
                self.obs = 5.0
                self.pos = pos

            def ReportFixture(self, fixture: b2Fixture, point, normal, fraction) -> float:
                if fixture.userData.type == ObjectType.Exit: #不对类型为Exit的物体进行检测
                    return fraction
                obs = ((point[0] - self.pos[0]) ** 2 + (point[1] - self.pos[1]) ** 2) ** 0.5
                self.obs = min(5.0, self.obs, obs)
                if test_mode:
                    print(fixture.userData,"###", obs)
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

    def __init__(self, env: b2World, new_x, new_y, new_width, new_height, display_level, object_type,
                 color=ColorRed):
        # new_x,new_y代表的是矩形墙的中心坐标
        self.body = env.CreateStaticBody(position=(new_x, new_y))
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
        x, y, width, height = transfer_to_render(self.body.position.x, self.body.position.y, self.width, self.height,
                                                 render_scale)
        self.pic = pyglet.shapes.Rectangle(x, y, width, height, self.color, batch, group=self.display_level)

    def delete(self):
        del(self)

    @property
    def getX(self):
        return self.body.position.x

    @property
    def getY(self):
        return self.body.position.y

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


