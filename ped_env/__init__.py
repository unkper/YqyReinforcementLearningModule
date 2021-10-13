import time

import Box2D as b2d
import pyglet

from ped_env.envs import PedsMoveEnv as Env
from ped_env.utils.maps import map_05, map_06, map_07


def HelloWorldProject():
    world = b2d.b2World()
    ground_body = world.CreateStaticBody(
        position=(0, -10),
        shapes=b2d.b2PolygonShape(box=(50, 10)),
    )
    body = world.CreateDynamicBody(position=(0, 4))

    box = body.CreatePolygonFixture(box=(1, 1),
                                    density=1,
                                    friction=0.3)
    timeStep = 1.0 / 60  # 时间步长，1/60秒
    vel_iters, pos_iters = 6, 2
    for i in range(600):  # 一共向前模拟60步，即总经过1秒
        world.Step(timeStep, vel_iters, pos_iters)

        # 清楚所有施加上的力，每次循环都是必须的
        world.ClearForces()

        # 打印输出物体的位置和角度
        print("Body Pos:{},Angle:{}".format(
            body.position,
            body.angle
        ))


def test1():
    import pyglet

    canvas = {}

    try:
        config = pyglet.gl.Config(double_buffer=True)
        window = pyglet.window.Window(1280, 720, resizable=True, config=config)
        window.set_minimum_size(640, 480)

        batch = pyglet.graphics.Batch()

        canvas[1] = pyglet.text.Label("Moo", x=10, y=10, batch=batch)

        @window.event
        def on_draw():
            window.clear()
            batch.draw()

        @window.event
        def on_key_press(symbol, modifiers):
            # As soon as a key is pressed, we delete the batch objects (all of them)
            for index in list(canvas):
                canvas[index].delete()
                del (canvas[index])

        pyglet.app.run()

    finally:
        window.close()


# Hello World Project
if __name__ == '__main__':
    import time
    import numpy as np

    # test1()
    person_num = 4
    env = Env(map_05, person_num, maxStep=10000)
    # print(obs)
    for epoch in range(100):
        starttime = time.time()
        step = 0
        obs = env.reset()
        is_done = [False]
        while not is_done[0]:
            action = np.random.random([person_num, 9])
            obs, reward, is_done, info = env.step(action)
            step += 1
            # env.render()
            # print(obs, reward, is_done)
        endtime = time.time()
        print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
              .format(env.listener.col_with_agent, env.listener.col_with_wall))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime, step / (endtime - starttime)))
