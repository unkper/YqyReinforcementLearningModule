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
from rl.env.puckworld_continous import PuckWorldEnv as Continous_PuckWorldEnv
from rl.env.gymEnvs import Pendulum
from rl.env.sisl import WaterWorld, MultiWalker
from rl.env.mpe import SimpleAdversary, SimpleSpeakerListener, SimplePusher, \
    SimpleSpread, SimpleWorldComm, SimpleCrypto, SimpleTag, SimpleReference
from rl.env.findTreasure import FindTreasureWrapper
from rl.agents.QAgents import DQNAgent,QAgent,DYNA_QAgent
from rl.agents.PDAgents import DDPGAgent
from rl.agents.MaddpgAgent import MADDPGAgent
from rl.utils.miscellaneous import learning_curve
from rl.utils.classes import OrnsteinUhlenbeckActionNoise

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

def test4(useEnv,envName):
    env = useEnv
    id = uuid1()
    print(env.observation_space)
    print(env.action_space)
    agent = MADDPGAgent(env,capacity=1e6,batch_size=1024,learning_rate=0.01,update_frequent=50,gamma=0.95)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_low=0.05,
                  max_episode_num=50000,
                  explore_episodes_percent=0.6
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, title="DQNAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", saveName=id.__str__())

def test5(useEnv,fileName,episode=3):
    env = useEnv
    agent = MADDPGAgent(env)
    agent.play(os.path.join("../data/models/",fileName),episode=episode,waitSecond=0.05)

if __name__ == '__main__':
    # envs = [(SimpleTag(),"SimpleTag"),(SimpleCrypto(),"SimpleCrypto"),(SimpleReference(),"SimpleReference")]
    # for item in envs:
    #     test4(item[0],item[1])
    env = SimpleTag(maxStep=125)
    test5(env,"2021_08_21_06_59_ddpg_SimpleTag_20000")
    # test4(env, "MultiWalker")


