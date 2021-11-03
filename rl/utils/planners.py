import copy
import datetime
import os
import pickle

from ped_env import map_05, map_06, map_07
from ped_env.envs import PedsMoveEnv
from ped_env.pathfinder import AStarController
from rl.utils.classes import Experience, Transition


class AStarPlanner:
    def __init__(self, env, capacity=1e6):
        '''
        该类利用env进行仿真模拟，然后将收集到的trans压入experience中
        :param env:
        :param experience:
        '''
        self.experience = Experience(capacity)
        self.init_time_str = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))

        def recode_trans(s, a, r, is_done, s1):
            trans = Transition(s, a, r, is_done, s1)
            self.experience.push(trans)
        env.planning = True
        self.controler = AStarController(env, recorder=recode_trans)

    def planning(self, episodes=1):
        self.controler.play(episodes, False)

    def clear_experience(self):
        self.experience.episodes.clear()

    def save_experience(self):
        file = open(os.path.join("./data",self.init_time_str+"_exp{}.pkl".format(self.experience.total_trans)),"wb")
        pickle.dump(self.experience,file,0)

    def load_experience(self, file):
        file = open(file,"rb")
        self.experience = pickle.load(file)

if __name__ == "__main__":
    env = PedsMoveEnv(map_05, person_num=30, group_size=(1, 6), maxStep=1000)
    planner = AStarPlanner(env)
    planner.planning(3000)
    planner.save_experience()
    # planner.load_experience("./data/2021_11_01_11_48_exp281.pkl")
    # print(planner.experience.sample(5))
