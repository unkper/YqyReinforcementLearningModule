import Box2D as b2d

from ped_env.envs import PedsMoveEnv as Env
from ped_env.envs import PedsMoveEnvFactory
from ped_env.pathfinder import AStarController
from ped_env.utils.maps import *
from rl.utils.classes import make_parallel_env
from ped_env.classes import PedsRLHandler, PedsRLHandlerRange

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

def test2():
    import time
    import numpy as np

    debug = False

    person_num = 40
    env = Env(map_10, person_num, group_size=(5, 5), maxStep=500, debug_mode=debug, random_init_mode=True)
    leader_num = env.agent_count

    for epoch in range(5000):
        starttime = time.time()
        step = 0
        obs = env.reset()
        is_done = [False]
        while not is_done[0]:
            if not debug:
                action = np.random.random([leader_num, 9])
            else:
                action = np.zeros([leader_num, 9])
                action[:, 0] = 1
            obs, reward, is_done, info = env.step(action)
            if debug:
                env.debug_step()
            step += env.frame_skipping
            env.render()

        endtime = time.time()
        print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
              .format(env.col_with_agent, env.col_with_wall))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime, step / (endtime - starttime)))

def test3():
    import time
    import numpy as np

    debug = False
    # test1()
    person_num = 8
    n_rol_counts = 4
    total_epochs = 4
    _env = Env(map_05, person_num, group_size=(1, 1), maxStep=500, test_mode=debug)
    parallel_envs = make_parallel_env(_env, n_rol_counts)
    leader_num = parallel_envs.agent_count
    for epoch in range(total_epochs):
        starttime = time.time()
        step = 0
        obs = parallel_envs.reset()
        is_done = np.array([[False]])
        while not is_done[0, 0]:
            if not debug:
                action = np.random.random([n_rol_counts, leader_num, 9])
            else:
                action = np.zeros([n_rol_counts, leader_num, 9])
                action[:, :, 0] = 1
            obs, reward, is_done, info = parallel_envs.step(action)
            step += _env.frame_skipping
            # parallel_envs.render()
            # print(obs, reward, is_done)
        endtime = time.time()
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime, step / (endtime - starttime)))

def test4():
    #使用连续动作空间的范例（随机策略）
    import time
    import numpy as np

    debug = False
    # test1()
    person_num = 40
    env = Env(map_02, person_num, group_size=(5, 5), maxStep=10000, discrete=False, test_mode=debug)
    leader_num = env.agent_count
    # print(obs)
    for epoch in range(1):
        starttime = time.time()
        step = 0
        obs = env.reset()
        is_done = [False]
        while not is_done[0]:
            if not debug:
                action = (np.random.random([leader_num, 2]) - 0.5) * 2
            else:
                action = np.zeros([leader_num, 2])
                action[:, 0] = 1
            obs, reward, is_done, info = env.step(action)
            # print(obs[0][9:11])
            if debug:
                env.debug_step()
            step += env.frame_skipping
            env.render()
            # print(obs, reward, is_done)
        endtime = time.time()
        print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
              .format(env.col_with_agent, env.col_with_wall))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime, step / (endtime - starttime)))

def test5():
    env = Env(map_05, 30, (5, 5))
    planner = AStarController(env)
    planner.play(5)

if __name__ == '__main__':
    test2()

    # import kdtree
    # points = []
    # for i in range(100):
    #     points.append([10 - i * 0.5, 0.3 * i])
    # tr = kdtree.create(points, dimensions=2)
    # print(kdtree.visualize(tr))