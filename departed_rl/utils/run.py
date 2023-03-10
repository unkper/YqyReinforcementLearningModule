import os,sys

curPath = os.path.abspath("../../" + os.path.curdir)
sys.path.append(curPath)

import argparse
import matplotlib.pyplot as plt

from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import *
from departed_rl.utils.planners import AStarPlanner

def func1(env, n_rol_threads, episodes, use_random=False, discrete=False):
    if use_random:
        print("生成随机策略经验!")
    planner = AStarPlanner(env, n_rol_threads=n_rol_threads, use_random_policy=use_random, discrete=discrete)
    planner.planning(episodes)
    planner.save_experience()

def func2():
    env = PedsMoveEnv(map_10, person_num=30, group_size=(5, 5), maxStep=1000)
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

#xvfb-run -a python run.py --map=map_10 --count=200 --p_num=32 --g_size=4 --thread=5
#xvfb-run -a python run.py --map=map_10 --count=50 --p_num=32 --g_size=4 --use_random=True --thread=5
#xvfb-run -a python run.py --map=map_05 --count=200 --thread=5
#xvfb-run -a python run.py --map=map_05 --count=50 --use_random=True --thread=5
#xvfb-run -a python run.py --map=map_02 --count=200 --use_random=True --random_init=True
#xvfb-run -a python run.py --map=map_05 --count=100 --use_random=True --threads=5
#xvfb-run -a python run.py --map=map_05 --count=1000 --use_random=False --random_init=True
#xvfb-run -a python run.py --map=map_06 --count=1000
#xvfb-run -a python run.py --map=map_06 --count=1000 --use_random=True
#xvfb-run -a python run.py --map=map_07 --count=1000 --p_num=40 --g_size=5
#xvfb-run -a python run.py --map=map_08 --count=1000
#python run.py --map=map_05 --count=60 --p_num=2 --g_size=1 --threads=2
#python run.py --map=map_05 --count=60 --p_num=2 --g_size=1 --threads=2 --use_random
if __name__ == '__main__':
    my_parser = argparse.ArgumentParser("Use A*/Random Policy generate experiences and save!")
    my_parser.add_argument('--map', default="map_12", type=str) #map_05
    my_parser.add_argument('--p_num', default=30, type=int)
    my_parser.add_argument('--g_size', default=5, type=int)
    my_parser.add_argument('--count', default=100, type=int)
    my_parser.add_argument('--max_step', default=1000, type=int)
    my_parser.add_argument('--threads', default=20, type=int) #20
    my_parser.add_argument('--use_random', default=False, type=bool)
    my_parser.add_argument('--discrete', default=True, type=bool)
    my_parser.add_argument('--random_init', default=True, type=bool)
    my_parser.add_argument('--frame_skip', default=8, type=int)


    args = my_parser.parse_args()
    env_dict = {
        "map_02": map_02,
        "map_05": map_05,
        "map_06": map_06,
        "map_10": map_10,
        "map_11": map_11,
        "map_12": map_12
    }

    maps = [env_dict[args.map]]
    for map in maps:
        env = PedsMoveEnv(terrain=map, person_num=args.p_num, group_size=(args.g_size, args.g_size),
                          maxStep=args.max_step, discrete=args.discrete, random_init_mode=args.random_init,
                          frame_skipping=args.frame_skip)
        func1(env, n_rol_threads=args.threads, episodes=args.count, use_random=args.use_random, discrete=args.discrete)