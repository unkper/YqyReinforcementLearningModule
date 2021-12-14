import os,sys

curPath = os.path.abspath("../../" + os.path.curdir)
sys.path.append(curPath)

import argparse
import matplotlib.pyplot as plt

from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import *
from rl.utils.planners import AStarPlanner

def func1(env, n_rol_threads, episodes, use_random=False):
    if use_random:
        print("生成随机策略经验!")
    planner = AStarPlanner(env, n_rol_threads=n_rol_threads, use_random_policy=use_random)
    planner.planning(episodes)
    planner.save_experience()

def func2():
    env = PedsMoveEnv(map_07, person_num=30, group_size=(5, 5), maxStep=1000)
    planner = AStarPlanner(env, n_rol_threads=1)
    planner.load_experience("./data/2021_12_09_14_39_map_07_exp1220.pkl")
    print(planner.experience.sample(5))

def func4():
    epsilon_high = 1.0
    epsilon_low = 0.1
    p = 0.99
    max_episode_num = 100
    x = [x for x in range(max_episode_num)]
    y = []
    for i in range(max_episode_num):
        epsilon = epsilon_low +( epsilon_high - epsilon_low) * np.power(np.e,-4/p*i/max_episode_num)
        y.append(epsilon)
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser("Use A*/Random Policy generate experiences and save!")
    my_parser.add_argument('--map', default="map_05", type=str)
    my_parser.add_argument('--p_num', default=30, type=int)
    my_parser.add_argument('--g_size', default=5, type=int)
    my_parser.add_argument('--count', default=500, type=int)
    my_parser.add_argument('--max_step', default=1000, type=int)
    my_parser.add_argument('--threads', default=20, type=int)
    my_parser.add_argument('--use_random', default=False, type=bool)
    args = my_parser.parse_args()
    env_dict = {
        "map_01": map_01,
        "map_02": map_02,
        "map_05": map_05,
        "map_07": map_07,
        "map_08": map_08,
        "map_09": map_09,
    }

    maps = [env_dict[args.map]]
    for map in maps:
        env = PedsMoveEnv(terrain=map, person_num=args.p_num, group_size=(args.g_size, args.g_size), maxStep=args.max_step)
        func1(env, n_rol_threads=args.threads, episodes=args.count, use_random=args.use_random)