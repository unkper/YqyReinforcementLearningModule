import datetime
import os
import sys
#为了Google colab挂载而加
curPath = os.path.abspath(os.path.curdir)
sys.path.append(curPath)

import gym
import torch

from uuid import uuid1
from gym.spaces import Discrete,Box
from rl.env.puckworld import PuckWorldEnv
from rl.env.puckworld_continous import PuckWorldEnv as Continous_PuckWorldEnv
from rl.env.findTreasure import FindTreasureWrapper
from rl.env.fireFighter import FireFighterWrapper
from rl.env.findGoal import FindGoalWrapper
from rl.agents.QAgents import DQNAgent,QAgent,DYNA_QAgent
from rl.agents.PDAgents import DDPGAgent
from rl.agents.MADDPGAgent import MADDPGAgent
from rl.utils.miscellaneous import learning_curve
from rl.utils.classes import OrnsteinUhlenbeckActionNoise

def test1():
    #env = gym.make("MountainCarContinuous-v0")
    env = Continous_PuckWorldEnv()
    print(env.observation_space)
    print(env.action_space)
    agent = DDPGAgent(env)
    id = uuid1()

    data = agent.learning(gamma=0.999,
                          decaying_epsilon=True,
                          max_episode_num=200)
    agent.save("PuckWorldEnv_Actor",agent.actor)
    agent.save("PuckWorldEnv_Critic",agent.critic)
    learning_curve(data, 2, 1, title="DQNAgent performance on {}".format(env.class_name()),
                   x_name="episodes", y_name="rewards of episode",saveName=id.__str__())

def test2():
    di = Discrete(7)
    di_samples = [di.sample() for i in range(10)]
    print(di_samples)
    bo = Box(-2.0,10000000,(4,4))
    print(bo.sample())

def test3():
    env = Continous_PuckWorldEnv()
    print(env.observation_space)
    print(env.action_space)
    agent = DDPGAgent(env)
    save_path = "../data/models/2021_08_04_05_30/PuckWorldEnv_Actor.pkl"
    agent.play(save_path,episode=20)

def test4():
    env = FindTreasureWrapper()
    id = uuid1()
    print(env.observation_space)
    print(env.action_space)
    agent = MADDPGAgent(env,capacity=10e6)
    data = agent.learning(gamma=0.999,
                  decaying_epsilon=True,
                  epsilon_low=0.1,
                  max_episode_num=50
                 )
    for i in range(agent.env.agent_count):
        agent.save("Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, title="DQNAgent performance on {}".format("FindTreasure"),
                   x_name="episodes", y_name="rewards of episode", saveName=id.__str__())

def test5():
    env = FindTreasureWrapper()
    agent = MADDPGAgent(env)
    agent.play("../data/models/2021_08_08_13_26",episode=5)

if __name__ == '__main__':
    test4()


