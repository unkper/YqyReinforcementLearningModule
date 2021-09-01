import time

import Box2D as b2d
import pyglet

from env.utils.drawer import MyDrawer
from env.envs import Box2DEnv1 as Env

def HelloWorldProject():
    world = b2d.b2World()
    ground_body = world.CreateStaticBody(
        position=(0,-10),
        shapes=b2d.b2PolygonShape(box=(50,10)),
    )
    body = world.CreateDynamicBody(position=(0,4))

    box = body.CreatePolygonFixture(box=(1,1),
                                    density=1,
                                    friction=0.3)
    timeStep = 1.0 / 60 #时间步长，1/60秒
    vel_iters, pos_iters = 6, 2
    for i in range(600):#一共向前模拟60步，即总经过1秒
        world.Step(timeStep, vel_iters, pos_iters)

        #清楚所有施加上的力，每次循环都是必须的
        world.ClearForces()

        #打印输出物体的位置和角度
        print("Body Pos:{},Angle:{}".format(
            body.position,
            body.angle
        ))

#Hello World Project
if __name__ == '__main__':
    env = Env()
    env.start()
    env.setup()
    pyglet.app.run()
