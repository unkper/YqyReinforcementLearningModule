import datetime
import os
import sys
import gym

#为了Google colab挂载而加
curPath = os.path.abspath("./" + os.path.curdir)
sys.path.append(curPath)

import ped_env.envs as my_env

from ped_env.utils.maps import map_02, map_01, map_04

from uuid import uuid1
from rl.utils.networks.dyna_network import dyna_model_network, dyna_q_network
from rl.env.puckworld import PuckWorldEnv
from rl.env.puckworld_continous import PuckWorldEnv as Continous_PuckWorldEnv
from rl.env.gymEnvs import Pendulum
from rl.env.moveBox import MoveBoxWrapper
from rl.env.sisl import WaterWorld, MultiWalker
from rl.env.mpe import SimpleAdversary, SimpleSpeakerListener, SimplePusher, \
    SimpleSpread, SimpleWorldComm, SimpleCrypto, SimpleTag, SimpleReference
from rl.env.findTreasure import FindTreasureWrapper
from rl.agents.QAgents import DQNAgent,QAgent
from rl.agents.QAgents import Deep_DYNA_QAgent as DDQAgent
from rl.agents.PDAgents import DDPGAgent
from rl.agents.MaddpgAgent import MADDPGAgent
from rl.agents.Matd3Agent import MATD3Agent
from rl.utils.miscellaneous import learning_curve
from rl.utils.classes import OrnsteinUhlenbeckActionNoise

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
    agent = MADDPGAgent(env,capacity=1e6,batch_size=1024,learning_rate=0.01
                        ,update_frequent=200,debug_log_frequent=100,gamma=0.95,tau=0.01,
                        env_name=envName)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.1,
                  max_episode_num=10000,
                  explore_episodes_percent=0.6
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, title="MADDPGAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", saveName=id.__str__())

def test5(useEnv,fileName,episode=3,AgentType=MADDPGAgent):
    env = useEnv
    agent = AgentType(env)
    agent.play(os.path.join("../data/models/",fileName),episode=episode,waitSecond=0.05)

def test6(env, envName, type=0):
    id = uuid1()
    agent_type_name = "DQNAgent" if type == 0 else "DDQAgent"
    agent = DQNAgent(env, gamma=0.99, learning_rate=1e-3, ddqn=True) if type == 0 else \
            DDQAgent(env, Q_net=dyna_q_network, model_net=dyna_model_network)
    data = agent.learning(max_episode_num=125)
    agent.save(agent.init_time_str + "_" + envName, agent_type_name, agent.behavior_Q if type==0 else agent.Q)
    learning_curve(data, 2, 1, title="QAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", saveName=id.__str__())
    if type == 1:
        data = []
        data.append(range(len(agent.loss_data)))
        data.append(agent.loss_data)
        learning_curve(data,title="loss data",x_name="episodes", y_name="loss value")

def test7(useEnv,envName):
    env = useEnv
    id = uuid1()
    print(env.observation_space)
    print(env.action_space)
    agent = MATD3Agent(env,capacity=1e6,batch_size=1024,learning_rate=0.01
                        ,update_frequent=50,debug_log_frequent=100,gamma=0.95,tau=0.01,
                        env_name=envName)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.05,
                  max_episode_num=5000,
                  explore_episodes_percent=0.6
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, title="MATD3Agent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", saveName=id.__str__())

if __name__ == '__main__':
    # envs = [(SimpleTag(),"SimpleTag"),(SimpleCrypto(),"SimpleCrypto"),(SimpleReference(),"SimpleReference")]
    # for item in envs:
    #     test4(item[0],item[1])
    envName = "PedsMoveEnv"
    env = my_env.PedsMoveEnv(terrain=map_02, person_num=4, maxStep=1500)
    #test4(env, envName)
    test7(env, envName)
    # test5(env, '2021_09_28_15_40_PedsMoveEnv', episode=5)

    # envName = "CartPole-v1"
    # env = gym.make(envName)
    # operation = input()
    # if operation == "1":
    #     test6(env, envName)
    # elif operation == "0":
    #     test5(env,"2021_09_22_15_55_CartPole-v1", episode=5, AgentType=DDQAgent)
    # elif operation == "2" or operation == "4":
    #     envName = "PedsMoveEnv"
    #     env = my_env.PedsMoveEnv(person_num=4, maxStep=3000)
    #     # envName = "SimpleSpread"
    #     # env = SimpleSpread(maxStep=50)
    #     if operation == "2":
    #         test4(env, envName)
    #     else:
    #         test5(env, '2021_09_26_18_30_PedsMoveEnv')
    # elif operation == "3":
    #     envName = "SimpleSpread"
    #     env = SimpleSpread()
    #     test4(env, envName)
    # else:
    #     test6(env, envName, 1)

