import datetime
import os
import sys
import gym

#为了Google colab挂载而加
curPath = os.path.abspath("./" + os.path.curdir)
sys.path.append(curPath)

import ped_env.envs as my_env

from ped_env.utils.maps import map_05, map_06, map_02

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
from rl.utils.networks.maddpg_network import MaddpgLstmActor, MaddpgLstmCritic
from rl.utils.miscellaneous import learning_curve
from rl.utils.classes import OrnsteinUhlenbeckActionNoise

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

def test5(useEnv,fileName,episode=3, AgentType=MADDPGAgent,
          actor_network=None, critic_network=None, hidden_dim=64):
    env = useEnv
    if not actor_network and not critic_network:
        agent = AgentType(env)
    else:
        agent = AgentType(env,actor_network=actor_network,critic_network=critic_network,hidden_dim=hidden_dim)
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

def test7(useEnv,envName,actor_network=None,critic_network=None,hidden_dim=64):
    env = useEnv
    id = uuid1()
    print(env.observation_space)
    print(env.action_space)
    agent = MATD3Agent(env,capacity=1e6,batch_size=1024,learning_rate=0.01
                        ,update_frequent=50,debug_log_frequent=100,gamma=0.99,tau=0.01,
                        env_name=envName,actor_network=actor_network,
                        critic_network=critic_network,hidden_dim=hidden_dim)#gamma=0.95
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.05,
                  max_episode_num=3000,
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
    env = my_env.PedsMoveEnv(terrain=map_05, person_num=8, maxStep=6000)
    # test4(env, envName)
    # test7(env, envName)
    # test7(env, envName, actor_network=MaddpgLstmActor, critic_network=MaddpgLstmCritic, hidden_dim=128)
    # test5(env, '2021_10_08_17_41_PedsMoveEnv', episode=5)
    test5(env, "2021_10_09_02_23_PedsMoveEnv", episode=10, AgentType=MATD3Agent, actor_network=MaddpgLstmActor, critic_network=MaddpgLstmCritic)

