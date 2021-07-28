import gym
import torch

from gym.spaces import Discrete,Box
from rl.agents.QAgents import DQNAgent,QAgent,Tabular_DYNA_QAgent
from rl.utils.miscellaneous import learning_curve
from rl.utils.classes import OrnsteinUhlenbeckActionNoise

def test1():
    env = gym.make("CartPole-v0")
    print(env.observation_space)
    print(env.action_space)
    agent = Tabular_DYNA_QAgent(env)

    data = agent.learning(gamma=0.99,
                          epsilon=0.9,
                          decaying_epsilon=True,
                          alpha=1e-3,
                          max_episode_num=200,
                          display=False)
    learning_curve(data, 2, 1, title="DQNAgent performance on {}".format(env.class_name()),
                   x_name="episodes", y_name="rewards of episode")
    save_path = agent.save_obj(agent.Q,"{}_{}")
    agent.play(save_path)

def test2():
    import numpy as np
    di = Discrete(7)
    di_samples = [di.sample() for i in range(10)]
    print(di_samples)
    bo = Box(-2.0,10000000,(4,4))
    print(bo.sample())

if __name__ == '__main__':
    test1()