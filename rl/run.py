import copy
import os
import sys

#为了Google colab挂载而加
curPath = os.path.abspath("../" + os.path.curdir)
sys.path.append(curPath)

import ped_env.envs as my_env

from ped_env.utils.maps import *

from uuid import uuid1
from rl.env.puckworld import PuckWorldEnv
from rl.env.puckworld_continous import PuckWorldEnv as Continous_PuckWorldEnv
from rl.env.sisl import WaterWorld, MultiWalker
from rl.env.mpe import SimpleAdversary, SimpleSpeakerListener, SimplePusher, \
    SimpleSpread, SimpleWorldComm, SimpleCrypto, SimpleTag, SimpleReference
from rl.agents.QAgents import DQNAgent,QAgent
from rl.agents.QAgents import Deep_DYNA_QAgent as DDQAgent
from rl.agents.G_MaddpgAgent import G_MADDPGAgent
from rl.agents.MaddpgAgent import MADDPGAgent
from rl.agents.Matd3Agent import MATD3Agent
from rl.utils.networks.dyna_network import dyna_model_network, dyna_q_network
from rl.utils.networks.maddpg_network import MLPNetworkActor, MLPNetwork_MACritic, \
                                             MaddpgLstmActor, MaddpgLstmCritic, G_MaddpgLstmModelNetwork
from rl.utils.miscellaneous import learning_curve
from rl.utils.planners import AStarPlanner


def test3(useEnv,envName,actor_network=None,critic_network=None,model_network=None,hidden_dim=64):
    env = useEnv
    id = uuid1()
    print(env.observation_space)
    print(env.action_space)
    planner_env = copy.deepcopy(useEnv)
    planner = AStarPlanner(planner_env)
    agent = G_MADDPGAgent(env,capacity=1e6,planner=planner,batch_size=1024,learning_rate=3e-4
                        ,update_frequent=200,debug_log_frequent=100,gamma=0.99,tau=0.01,
                        env_name=envName,actor_network=actor_network,critic_network=critic_network,
                        model_network=model_network,hidden_dim=hidden_dim)
    agent.opt_init(1500)
    print("opt_init finished!")
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=0.6,
                  epsilon_low=0.05,
                  max_episode_num=1500,
                  explore_episodes_percent=0.85
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, title="G_MADDPGAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName=id.__str__())

def test4(useEnv,envName,n_rol_threads=8,actor_network=None,critic_network=None,hidden_dim=64):
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = MADDPGAgent(env,n_rol_threads=n_rol_threads,capacity=1e6,batch_size=1024,learning_rate=3e-4
                        ,update_frequent=5,debug_log_frequent=20,gamma=0.99,tau=0.01,
                        env_name=envName,actor_network=actor_network,
                        critic_network=critic_network,hidden_dim=hidden_dim)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.01,
                  max_episode_num=3200,
                  explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, title="MADDPGAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName="maddpg")

def test5(useEnv,fileName,episode=3, AgentType=MADDPGAgent,
          actor_network=None, critic_network=None, model_network=None, hidden_dim=64):
    env = useEnv
    if not actor_network and not critic_network:
        agent = AgentType(env)
    elif model_network != None:
        agent = AgentType(env,actor_network=actor_network,critic_network=critic_network,model_network=model_network,hidden_dim=hidden_dim)
    else:
        agent = AgentType(env,actor_network=actor_network,critic_network=critic_network,hidden_dim=hidden_dim)
    agent.play(os.path.join("../data/models/",fileName,"model"),episode=episode,waitSecond=0.05)

def test6(env, envName, type=0):
    id = uuid1()
    agent_type_name = "DQNAgent" if type == 0 else "DDQAgent"
    agent = DQNAgent(env, gamma=0.99, learning_rate=1e-3, ddqn=True) if type == 0 else \
            DDQAgent(env, Q_net=dyna_q_network, model_net=dyna_model_network)
    data = agent.learning(max_episode_num=125)
    agent.save(agent.init_time_str + "_" + envName, agent_type_name, agent.behavior_Q if type==0 else agent.Q)
    learning_curve(data, 2, 1, title="QAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir,  saveName=id.__str__())
    if type == 1:
        data = []
        data.append(range(len(agent.loss_data)))
        data.append(agent.loss_data)
        learning_curve(data,title="loss data",x_name="episodes", y_name="loss value")

def test7(useEnv,envName,n_rol_threads=8,actor_network=None,critic_network=None,hidden_dim=64):
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = MATD3Agent(env,n_rol_threads=n_rol_threads,capacity=1e6,batch_size=1024,learning_rate=3e-4
                        ,update_frequent=5,debug_log_frequent=20,gamma=0.99,tau=0.01,
                        env_name=envName,actor_network=actor_network,
                        critic_network=critic_network,hidden_dim=hidden_dim)#gamma=0.95
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.01,
                  max_episode_num=2,
                  explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, step_index=0, title="MATD3Agent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName="matd3")

if __name__ == '__main__':
    # envs = [(SimpleWorldComm(maxStep=250),"SimpleWorldComm"),(SimpleCrypto(),"SimpleCrypto"),(SimpleReference(),"SimpleReference")]
    # for env in envs:
    #     test7(env[0], env[1], actor_network=MaddpgLstmActor, critic_network=MaddpgLstmCritic, hidden_dim=128)
    # save_file_names = ['2021_10_31_20_21_SimpleWorldComm', '2021_10_31_21_30_SimpleCrypto',
    #                        '2021_10_31_21_53_SimpleReference']
    # for i in range(len(envs)):
    #     test5(envs[i][0], save_file_names[i], episode=5, AgentType=MATD3Agent, actor_network=MaddpgLstmActor, critic_network=MaddpgLstmCritic)
    # for item in envs:
    #     test4(item[0],item[1])
    envName = "PedsMoveEnv"
    maps = [map_05, map_06, map_08]

    env = my_env.PedsMoveEnv(terrain=map_05, person_num=30, group_size=(5,5), maxStep=20000)
    test7(env, envName, n_rol_threads=8, actor_network=MLPNetworkActor, critic_network=MLPNetwork_MACritic,
                hidden_dim=128)
    # for i in range(2):
    #     test7(env, envName, n_rol_threads=8, actor_network=MLPNetworkActor, critic_network=MLPNetwork_MACritic, hidden_dim=128)
    # test4(env, envName, n_rol_threads=16, actor_network=MLPNetworkActor, critic_network=MLPNetwork_MACritic, hidden_dim=128)

    # test3(env, envName, actor_network=MaddpgLstmActor, critic_network=MaddpgLstmCritic,model_network=G_MaddpgLstmModelNetwork, hidden_dim=128)

    # test5(env, '2021_11_11_13_36_PedsMoveEnv', episode=5, AgentType=MATD3Agent, actor_network=MLPNetworkActor,critic_network=MLPNetwork_MACritic) #model_network=G_MaddpgLstmModelNetwork)

    # for map in maps:
    #     env = my_env.PedsMoveEnv(terrain=map, person_num=30 if map != map_hard_obj else 32, group_size=(5, 5) if map != map_hard_obj else (4, 4), maxStep=20000)
    #     test7(env, envName, n_rol_threads=16, actor_network=MLPNetworkActor, critic_network=MLPNetwork_MACritic,
    #           hidden_dim=128)

    # save_file_names = ['2021_10_31_23_36_PedsMoveEnv', '2021_11_01_01_10_PedsMoveEnv',
    #                        '2021_11_01_03_32_PedsMoveEnv']
    # for map,save_name in zip(maps,save_file_names):
    #     env = my_env.PedsMoveEnv(terrain=map, person_num=8, group_size=(1,1), maxStep=3000)
    #     test5(env, save_name, episode=5, AgentType=MATD3Agent, actor_network=MaddpgLstmActor, critic_network=MaddpgLstmCritic)


