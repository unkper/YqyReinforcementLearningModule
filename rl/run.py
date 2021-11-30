import copy
import os
import sys

#为了Google colab挂载而加
curPath = os.path.abspath("../" + os.path.curdir)
sys.path.append(curPath)

import ped_env.envs as my_env

from ped_env.utils.maps import *
from rl.env.mpe import SimpleSpread

from uuid import uuid1
from rl.agents.QAgents import DQNAgent,QAgent
from rl.agents.QAgents import Deep_DYNA_QAgent as DDQAgent
from rl.agents.G_MaddpgAgent import G_MADDPGAgent
from rl.agents.MaddpgAgent import MADDPGAgent
from rl.agents.Matd3Agent import MATD3Agent
from rl.utils.networks.dyna_network import dyna_model_network, dyna_q_network
from rl.utils.networks.maddpg_network import MLPNetworkActor, MLPNetworkCritic, DoubleQNetworkCritic, \
                                             MaddpgLstmActor, MaddpgLstmCritic, G_MaddpgLstmModelNetwork, MLPModelNetwork
from rl.utils.miscellaneous import learning_curve

from rl.config import *

def test3(useEnv,envName,config:Config):
    env = useEnv
    id = uuid1()
    print(env.observation_space)
    print(env.action_space)
    agent = G_MADDPGAgent(env, n_rol_threads=config.n_rol_threads, capacity=config.capacity, batch_size=config.batch_size, learning_rate=config.learning_rate
                          , update_frequent=config.update_frequent, debug_log_frequent=config.debug_log_frequent, gamma=config.gamma, tau=config.tau,
                          env_name=envName, actor_network=config.actor_network, critic_network=config.critic_network,
                          actor_hidden_dim=config.actor_hidden_dim, critic_hidden_dim=config.critic_hidden_dim,
                          real_ratio=config.real_ratio, n_steps_train=config.n_steps_train, model_train_freq=config.model_train_freq,
                          n_steps_model=config.n_steps_model, model_batch_size=config.model_batch_size,
                          model_hidden_dim=config.model_hidden_dim, rollout_epoch_range=config.rollout_epoch_range) #training_count = update_fre/ num_train_repeat 该值越小训练时间越长
                                                                                   #model_lr = 0.001 0.01会出现严重的不收敛 1episode=(,1472) model_train_freq=400 不抽取经验池中所有而是一个batch_size
                                                                                  #模型在50%的时候预测精度开始严重下降
    data = agent.learning(
                    decaying_epsilon=True,
                    epsilon_high=1.0,
                    epsilon_low=0.01,
                    max_episode_num=config.max_episode,
                    explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, step_index=0, title="G_MADDPGAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName=id.__str__())

def test4(useEnv,envName,config:Config):
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = MADDPGAgent(env, n_rol_threads=config.n_rol_threads, capacity=config.capacity, batch_size=config.batch_size,
                        learning_rate=config.learning_rate, update_frequent=config.update_frequent,
                        debug_log_frequent=config.debug_log_frequent, gamma=config.gamma, tau=config.tau,
                        env_name=envName, actor_network=config.actor_network, n_steps_train=config.n_steps_train,
                        critic_network=config.critic_network, actor_hidden_dim=config.actor_hidden_dim, critic_hidden_dim=config.critic_hidden_dim)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.01,
                  max_episode_num=config.max_episode,
                  explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, step_index=0, title="MADDPGAgent performance on {}".format(envName),
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

def test7(useEnv,envName,n_rol_threads=8,max_episode=1600,actor_network=None,critic_network=None,hidden_dim=64):
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = MATD3Agent(env,n_rol_threads=n_rol_threads,capacity=1e6,batch_size=1024,learning_rate=3e-4
                        ,update_frequent=5,debug_log_frequent=20,gamma=0.99,tau=0.01,K = 2,
                        env_name=envName,actor_network=actor_network,
                        critic_network=critic_network,hidden_dim=hidden_dim)#gamma=0.95
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.01,
                  max_episode_num=max_episode,
                  explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor)
    learning_curve(data, 2, 1, step_index=0, title="MATD3Agent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName="matd3")

if __name__ == '__main__':
    #envName = "PedsMoveEnv"
    maps = [map_02, map_03]

    #env = my_env.PedsMoveEnv(terrain=map_05, person_num=6, group_size=(1, 1), maxStep=20000)\
    envName = "SimpleSpread"
    env = SimpleSpread()
    config = MPEConfig(max_episode=400,n_rol_threads=10)
    # test4(env, envName, n_rol_threads=20, max_episode=2000, actor_network=MLPNetworkActor,
    #     critic_network=MLPNetworkCritic, actor_hidden_dim=128, critic_hidden_dim=256)
    test3(env, envName, config=config)

    # for batch_num in range(20):
    #     for map in maps:
    #         env = my_env.PedsMoveEnv(terrain=map, person_num=30, group_size=(5, 5), maxStep=20000)
    #         test3(env, envName, N=64, n_rol_threads=16, max_episode=1600, actor_network=MLPNetworkActor, critic_network=MLPNetworkCritic,
    #               model_network=MLPModelNetwork, hidden_dim=128)
    #         test4(env, envName, n_rol_threads=16, max_episode=1600, actor_network=MLPNetworkActor, critic_network=MLPNetworkCritic, hidden_dim=128)

    # for map in maps:
    #     env = my_env.PedsMoveEnv(terrain=map, person_num=30, group_size=(5, 5), maxStep=20000)
    #     test3(env, envName, n_rol_threads=16, max_episode=3200, actor_network=MLPNetworkActor, critic_network=MLPNetworkCritic,
    #           model_network=MLPModelNetwork, hidden_dim=128)
    #     test7(env, envName, n_rol_threads=16, max_episode=3200, actor_network=MLPNetworkActor, critic_network=DoubleQNetworkCritic, hidden_dim=128)
    #     test4(env, envName, n_rol_threads=16, max_episode=3200, actor_network=MLPNetworkActor, critic_network=MLPNetworkCritic, hidden_dim=128)

    # map = map_05
    # env = my_env.PedsMoveEnv(terrain=map, person_num=30, group_size=(5, 5), maxStep=20000)
    # test3(env, envName, n_rol_threads=16, actor_network=MLPNetworkActor, critic_network=MLPNetworkCritic,model_network=MLPModelNetwork, hidden_dim=128)

    # env = my_env.PedsMoveEnv(terrain=map_05, person_num=30, group_size=(5,5), maxStep=20000)
    # test5(env, '2021_11_12_22_32_PedsMoveEnv', episode=5, AgentType=MADDPGAgent, actor_network=MLPNetworkActor, critic_network=MLPNetwork_MACritic) #model_network=G_MaddpgLstmModelNetwork)

