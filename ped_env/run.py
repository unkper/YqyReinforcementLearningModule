import os.path
import pprint
import random

import Box2D as b2d

from ped_env.envs import PedsMoveEnv as Env
from ped_env.pathfinder import AStarPolicy
from ped_env.utils.maps import *
from departed_rl.utils.classes import make_parallel_env, PedsMoveInfoDataHandler
from ped_env.mdp import PedsRLHandler, PedsRLHandlerWithPlanner, PedsVisionRLHandler, PedsRLHandlerWithForce
#from rl_platform.tianshou_case.utils.wrappers import FrameStackWrapper


# 使用随机策略来前往目的地
def test2():
    import time

    debug = True

    person_num = 10
    env = Env("map_11", person_num, group_size=(1, 1), frame_skipping=8, maxStep=10000, debug_mode=debug,
              random_init_mode=True, person_handler=None, with_force=True)
    leader_num = env.agent_count
    #handler = PedsMoveInfoDataHandler(env.terrain, env.agent_count)

    for epoch in range(5):
        start_time = time.time()
        step = 0
        env.reset()
        is_done = {env.agents[0]: False}

        def get_single_action(agent):
            return env.action_space(agent).sample()

        while not all(is_done.values()):
            action = {agent: get_single_action(agent) for agent in env.agents}
            obs, reward, is_done, truncated, info = env.step(action)
            # pprint.pprint(obs)
            if debug:
                env.debug_step()
            step += env.frame_skipping
            env.render(ratio=1)
            #pprint.pprint(env.not_arrived_peds)
        endtime = time.time()
        print("智能体与智能体碰撞次数为{},与墙碰撞次数为{}!"
              .format(env.collide_agents_count, env.collide_wall_count))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - start_time,
                                                                             step / (endtime - start_time)))
    #handler.save("./")


def test3():
    import time
    import numpy as np

    debug = False
    # test1()
    person_num = 8
    n_rol_counts = 4
    total_epochs = 4
    _env = Env("map_10", person_num, group_size=(1, 1), maxStep=500)
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
    env = Env("map_11", person_num, group_size=(1, 1), maxStep=10000, discrete=False)
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
              .format(env.collide_agents_count, env.collide_wall_count))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime,
                                                                             step / (endtime - starttime)))


# 使用A*策略来前往目的地
def test5():
    import time
    import numpy as np

    debug = False

    person_num = 32
    env = Env("map_12", person_num, group_size=(4, 4), frame_skipping=8, maxStep=300, debug_mode=debug,
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
              .format(env.collide_agents_count, env.collide_wall_count))
        print("所有智能体在{}步后离开环境,离开用时为{},两者比值为{}!".format(step, endtime - starttime,
                                                                             step / (endtime - starttime)))


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def test_wrapper_api(debug=False):
    import time
    import ped_env.settings as setting

    person_num = 20
    env = Env(map_10, person_num, group_size=(1, 1), frame_skipping=8, maxStep=10000, debug_mode=False,
              random_init_mode=True, person_handler=PedsVisionRLHandler)
    # env = Env(map_simple, person_num, group_size=(1, 1), frame_skipping=8, maxStep=10000, debug_mode=False,
    #           random_init_mode=True, person_handler=PedsRLHandlerWithForce)

    env = FrameStackWrapper(env)

    for epoch in range(1):
        start_time = time.time()
        step = 0
        is_done = {'0': False}
        env.reset(None, None)

        def get_single_action(agent):
            return env.action_space(agent).sample()

        obs_arr = []

        while not all(is_done.values()):
            action = {agent: get_single_action(agent) for agent in env.agents}
            obs, reward, is_done, truncated, info = env.step(action)
            # pprint.pprint(reward)
            obs_arr.append(obs['0'][0])
            # pprint.pprint(obs)
            if debug:
                env.debug_step()
            step += env.frame_skipping
            #env.render()

        if debug and isinstance(env.venv.person_handler, PedsVisionRLHandler):
            save_video(obs_arr)


def save_video(obs_arr, name="animation"):
    """
    obs_arr是一系列帧堆叠成的数组
    """
    fig, ax = plt.subplots()
    now_obs_idx = 0

    def update(frame):
        nonlocal now_obs_idx, obs_arr
        ax.clear()

        ax.imshow(obs_arr[now_obs_idx])
        # ax.imshow(np.transpose(obs_arr[now_obs_idx], (1, 2, 0)))
        now_obs_idx += 1
        now_obs_idx %= len(obs_arr)
        # 隐藏坐标轴
        ax.axis('off')

    ani = FuncAnimation(fig, update)
    path = './{}.mp4'.format(name)
    if os.path.exists(path):
        os.remove(path)
    ani.save(path, writer='ffmpeg')


if __name__ == '__main__':
    # HelloWorldProject()
    test2()
    # test_wrapper_api(debug=True)

    # import kdtree
    # points = []
    # for i in range(100):
    #     points.append([10 - i * 0.5, 0.3 * i])
    # tr = kdtree.create(points, dimensions=2)
    # print(kdtree.visualize(tr))
