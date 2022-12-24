import pprint
import random

import Box2D as b2d
from gym.spaces import Discrete

from ped_env.utils.new_map import NewMap
from ped_env.envs import PedsMoveEnv as Env
from ped_env.pathfinder import AStarController, AStarPolicy
from ped_env.utils.maps import *
from rl.utils.classes import make_parallel_env, PedsMoveInfoDataHandler
from ped_env.mdp import PedsRLHandler, PedsRLHandlerWithPlanner


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


# 使用随机策略来前往目的地
def test2():
    import time
    import numpy as np

    debug = False

    person_num = 100
    env = Env(map_10, person_num, group_size=(1, 1), frame_skipping=8, maxStep=10000, debug_mode=debug,
              random_init_mode=True)
    leader_num = env.agent_count
    handler = PedsMoveInfoDataHandler(env.terrain, env.agent_count)

    for epoch in range(5):
        starttime = time.time()
        step = 0
        obs = env.reset()
        is_done = {env.agents[0]: False}
        speed_action = random.sample([1], 1)[0]

        def get_single_action(agent):
            return env.action_space(agent).sample()
            # return 18

        while not all(is_done.values()):
            # if not debug:
            #     action = np.random.random([leader_num, 9])
            # else:
            #     action = np.zeros([leader_num, 9])
            #     action[:, 0] = 1
            # action = {agent: env.action_space(agent).sample() for agent in env.agents}
            action = {agent: get_single_action(agent) for agent in env.agents}
            obs, reward, is_done, truncated, info = env.step(action)
            # pprint.pprint(obs)
            if debug:
                env.debug_step()
            step += env.frame_skipping
            env.render()
        endtime = time.time()
        print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
              .format(env.collision_between_agents, env.collide_wall))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime,
                                                                             step / (endtime - starttime)))
    handler.save("./")


def test3():
    import time
    import numpy as np

    debug = False
    # test1()
    person_num = 8
    n_rol_counts = 4
    total_epochs = 4
    _env = Env(map_05, person_num, group_size=(1, 1), maxStep=500)
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
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime,
                                                                             step / (endtime - starttime)))


def test4():
    # 使用连续动作空间的范例（随机策略）
    import time
    import numpy as np

    debug = False
    # test1()
    person_num = 40
    env = Env(map_11, person_num, group_size=(1, 1), maxStep=10000, discrete=False)
    leader_num = env.agent_count
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
              .format(env.collision_between_agents, env.collide_wall))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime,
                                                                             step / (endtime - starttime)))


# 使用A*策略来前往目的地
def test5():
    import time
    import numpy as np

    debug = False

    person_num = 32
    env = Env(map_12, person_num, group_size=(4, 4), frame_skipping=8, maxStep=300, debug_mode=debug,
              random_init_mode=True)
    leader_num = env.agent_count
    policy = AStarPolicy(env.terrain)

    for epoch in range(20):
        starttime = time.time()
        step = 0
        obs = env.reset()
        is_done = [False]
        while not is_done[0]:
            action = policy.step(obs)
            for i in range(1):
                obs, reward, is_done, info = env.step(action)
                env.render()
                if is_done[0]:
                    break
                if debug:
                    env.debug_step()
                step += env.frame_skipping
        endtime = time.time()
        print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
              .format(env.collision_between_agents, env.collide_wall))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime,
                                                                             step / (endtime - starttime)))


if __name__ == '__main__':
    # HelloWorldProject()
    test2()

    # import kdtree
    # points = []
    # for i in range(100):
    #     points.append([10 - i * 0.5, 0.3 * i])
    # tr = kdtree.create(points, dimensions=2)
    # print(kdtree.visualize(tr))
