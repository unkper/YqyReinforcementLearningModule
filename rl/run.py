import copy
import os
import sys

#为了Google colab挂载而加
curPath = os.path.abspath("../" + os.path.curdir)
sys.path.append(curPath)
sys.path.append(os.path.join(curPath, './rl/env'))
sys.path.append(os.path.join(curPath, './rl/env/multiagent_particle_envs'))

import ped_env.envs as my_env

from ped_env.utils.maps import *
from rl.env.mpe import SimpleSpread, SimpleTag, SimpleSpread_v3

from rl.agents.G_MaddpgAgent import G_MADDPGAgent
from rl.agents.MaddpgAgent import MADDPGAgent
from rl.agents.Matd3Agent import MATD3Agent
from rl.agents.G_Matd3Agent import G_MATD3Agent
from rl.utils.miscellaneous import learning_curve, save_parameter_setting

from rl.config import *

def test1(useEnv, envName, config:Config):
    config.update_parameter("matd3")
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = MATD3Agent(env,n_rol_threads=config.n_rol_threads,capacity=config.capacity,batch_size=config.batch_size,
                       learning_rate=config.learning_rate,update_frequent=config.update_frequent,debug_log_frequent=config.debug_log_frequent,
                       gamma=config.gamma,tau=config.tau,K = config.K,actor_network=config.actor_network,
                       critic_network=config.critic_network,actor_hidden_dim=config.actor_hidden_dim,
                       critic_hidden_dim=config.critic_hidden_dim,n_steps_train=config.n_steps_train,
                       env_name=envName)#gamma=0.95
    save_parameter_setting(agent.log_dir,"matd3",config)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.01,
                  max_episode_num=config.max_episode,
                  explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor, config.max_episode)
    learning_curve(data, 2, 1, step_index=0, title="MATD3Agent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName="matd3")

def test2(useEnv, envName, config:Config):
    config.update_parameter("g_matd3")
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = G_MATD3Agent(env,n_rol_threads=config.n_rol_threads,capacity=config.capacity,batch_size=config.batch_size,
                       learning_rate=config.learning_rate,update_frequent=config.update_frequent,debug_log_frequent=config.debug_log_frequent,
                       gamma=config.gamma,tau=config.tau,K = config.K,actor_network=config.actor_network,
                       critic_network=config.critic_network,actor_hidden_dim=config.actor_hidden_dim, model_hidden_dim=config.model_hidden_dim,
                       critic_hidden_dim=config.critic_hidden_dim,n_steps_train=config.n_steps_train,
                       env_name=envName,init_train_steps=config.init_train_steps,
                       network_size=config.network_size,elite_size=config.elite_size, use_decay=config.use_decay,
                       model_batch_size=config.model_batch_size, model_train_freq=config.model_train_freq, n_steps_model=config.n_steps_model,
                       rollout_length_range=config.rollout_length_range,rollout_epoch_range=config.rollout_epoch_range,
                       rollout_batch_size=config.rollout_batch_size,real_ratio=config.real_ratio,
                       model_retain_epochs=config.model_retain_epochs)#gamma=0.95
    save_parameter_setting(agent.log_dir,"g_matd3",config)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.01,
                  max_episode_num=config.max_episode,
                  explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor, config.max_episode)
    learning_curve(data, 2, 1, step_index=0, title="G_MATD3Agent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName="g_matd3")

def test3(useEnv,envName,config:Config):
    config.update_parameter("g_maddpg")
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = G_MADDPGAgent(env, capacity=config.capacity, n_rol_threads=config.n_rol_threads, batch_size=config.batch_size, learning_rate=config.learning_rate,
                          update_frequent=config.update_frequent, debug_log_frequent=config.debug_log_frequent, gamma=config.gamma, tau=config.tau,
                          actor_network=config.actor_network, critic_network=config.critic_network,
                          actor_hidden_dim=config.actor_hidden_dim, critic_hidden_dim=config.critic_hidden_dim, model_hidden_dim=config.model_hidden_dim, env_name=envName,
                          init_train_steps=config.init_train_steps,
                          network_size=config.network_size,elite_size=config.elite_size, use_decay=config.use_decay,
                          model_batch_size=config.model_batch_size, model_train_freq=config.model_train_freq, n_steps_model=config.n_steps_model,
                          rollout_length_range=config.rollout_length_range,rollout_epoch_range=config.rollout_epoch_range,
                          rollout_batch_size=config.rollout_batch_size,real_ratio=config.real_ratio,
                          model_retain_epochs=config.model_retain_epochs, n_steps_train=config.n_steps_train) #training_count = update_fre/ num_train_repeat 该值越小训练时间越长
                                                                                                              #model_lr = 0.001 0.01会出现严重的不收敛 1episode=(,1472) model_train_freq=400 不抽取经验池中所有而是一个batch_size
                                                                                                              #模型在50%的时候预测精度开始严重下降
    save_parameter_setting(agent.log_dir,"g_maddpg",config)
    data = agent.learning(
                    decaying_epsilon=True,
                    epsilon_high=1.0,
                    epsilon_low=0.01,
                    max_episode_num=config.max_episode,
                    explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor, config.max_episode)
    learning_curve(data, 2, 1, step_index=0, title="G_MADDPGAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName="G_maddpg")

def test4(useEnv,envName,config:Config):
    config.update_parameter("maddpg")
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = MADDPGAgent(env, n_rol_threads=config.n_rol_threads, capacity=config.capacity, batch_size=config.batch_size,
                        learning_rate=config.learning_rate, update_frequent=config.update_frequent,
                        debug_log_frequent=config.debug_log_frequent, gamma=config.gamma, tau=config.tau,
                        actor_network=config.actor_network, critic_network=config.critic_network,
                        actor_hidden_dim=config.actor_hidden_dim, critic_hidden_dim=config.critic_hidden_dim,
                        env_name=envName, n_steps_train=config.n_steps_train)
    save_parameter_setting(agent.log_dir,"maddpg",config)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,
                  epsilon_low=0.01,
                  max_episode_num=config.max_episode,
                  explore_episodes_percent=1.0
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.init_time_str + "_" + envName,"Actor{}".format(i),agent.agents[i].actor, config.max_episode)
    learning_curve(data, 2, 1, step_index=0, title="MADDPGAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName="maddpg")

def eval(useEnv,fileName,load_E,episode=5, AgentType=MATD3Agent,
          config:Config=None):
    env = useEnv
    agent = AgentType(env,actor_network=config.actor_network, critic_network=config.critic_network,
                        actor_hidden_dim=config.actor_hidden_dim, critic_hidden_dim=config.critic_hidden_dim)
    agent.play(os.path.join("../data/models/",fileName,"model"), load_E, episode=episode, waitSecond=0.05)

def testEnv():
    env = SimpleSpread_v3()
    state = env.reset()
    is_done = [False]
    for i in range(1000):
        while sum(is_done) == 0:
            obs, reward, is_done, info = env.step(np.random.randn(3,2))
            env.render()
            print(obs)
            print(reward)

if __name__ == '__main__':
    envs = [#("PedsMoveEnv",
            # my_env.PedsMoveEnv(terrain=map_02, person_num=24, group_size=(4, 4), maxStep=20000, discrete=False)),
            ("PedsMoveEnv",
            my_env.PedsMoveEnv(terrain=map_05, person_num=24, group_size=(4, 4), maxStep=10000)),
            ("PedsMoveEnv",
            my_env.PedsMoveEnv(terrain=map_06, person_num=24, group_size=(4, 4), maxStep=10000)),
            ]
    config = PedsMoveConfig(n_rol_threads=10, max_episode=100)
    # eval(envs[0][1], "2021_12_06_20_38_PedsMoveEnv", 50, config=config)
    # envs = [("SimpleSpread", SimpleSpread_v3())]
    # config = MPEConfig(max_episode=5000, n_rol_threads=1)

    for i in range(1):
        for envName, env in envs:
            # config.max_episode = 500
            # print("Matd3 10steps gradient descent training!")
            # test1(env, envName, config=config) #Matd3 10Step
            # config.n_steps_train = 1
            # print("Matd3 1steps gradient descent training!")
            # test1(env, envName, config=config) #Matd3 1Step
            config.max_episode = 200
            config.n_steps_train = 10
            print("MBPO-Matd3 gradient descent training!")
            test2(env, envName, config=config) #Model-Based Matd3 10Step


    # for batch_num in range(20):
    #     for map in maps:
    #         env = my_env.PedsMoveEnv(terrain=map, person_num=30, group_size=(5, 5), maxStep=20000)
    #         test3(env, envName, N=64, n_rol_threads=16, max_episode=1600, actor_network=MLPNetworkActor, critic_network=MLPNetworkCritic,
    #               model_network=MLPModelNetwork, hidden_dim=128)
    #         test4(env, envName, n_rol_threads=16, max_episode=1600, actor_network=MLPNetworkActor, critic_network=MLPNetworkCritic, hidden_dim=128)

    # env = my_env.PedsMoveEnv(terrain=map_05, person_num=30, group_size=(5,5), maxStep=20000)
    # test5(env, '2021_11_12_22_32_PedsMoveEnv', episode=5, AgentType=MADDPGAgent, actor_network=MLPNetworkActor, critic_network=MLPNetwork_MACritic) #model_network=G_MaddpgLstmModelNetwork)

