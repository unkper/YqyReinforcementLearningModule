import multiprocessing
import os
import sys

import torch

curPath = os.path.abspath("../" + os.path.curdir)
sys.path.append(curPath)
sys.path.append(os.path.join(curPath, './rl/env'))
sys.path.append(os.path.join(curPath, './rl/env/multiagent_particle_envs'))

import ped_env.envs as my_env
import argparse

from multiprocessing import Process
from ped_env.utils.maps import *
from rl.env.mpe import SimpleSpread, SimpleTag, SimpleSpread_v3
from rl.utils.planners import init_offline_train
from rl.agents.G_MaddpgAgent import G_MADDPGAgent
from rl.agents.MaddpgAgent import MADDPGAgent
from rl.agents.Matd3Agent import MATD3Agent
from rl.agents.MAMBPOAgent import MAMBPOAgent
from rl.utils.miscellaneous import learning_curve, save_parameter_setting

from rl.config import *

def test1(useEnv, envName, config:Config, lock=None):
    config.update_parameter("MATD3")
    offline_data = useEnv.terrain.name + "_exp"
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = MATD3Agent(env,n_rol_threads=config.n_rol_threads,capacity=config.capacity,batch_size=config.batch_size,
                       learning_rate=config.learning_rate,update_frequent=config.update_frequent,debug_log_frequent=config.debug_log_frequent,
                       gamma=config.gamma,tau=config.tau,K = config.K,log_dir=config.log_dir,actor_network=config.actor_network,
                       critic_network=config.critic_network,actor_hidden_dim=config.actor_hidden_dim,
                       critic_hidden_dim=config.critic_hidden_dim,n_steps_train=config.n_steps_train,
                       env_name=envName,batch_size_d=config.batch_size_d)#gamma=0.95
    save_parameter_setting(agent.log_dir,"matd3",config)
    if config.use_init_bc:
        init_offline_train(agent, "./utils/data/exp/{}/experience.pkl".format(offline_data), config.init_bc_steps, lock=lock)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0, # if not config.use_init_bc else 0.1
                  epsilon_low=0.01,
                  max_episode_num=config.max_episode,
                  explore_episodes_percent=1.0,
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.log_dir,"Actor{}".format(i),agent.agents[i].actor, config.max_episode)

def test2(useEnv, envName, config:Config, debug=False, lock=None):
    if not debug:
        offline_data = useEnv.terrain.name + "_exp"
        random_data = useEnv.terrain.name + "_exp_random"
    config.update_parameter("GD_MAMBPO" if config.use_init_bc else "MAMBPO")
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    agent = MAMBPOAgent(env, n_rol_threads=config.n_rol_threads, capacity=config.capacity, batch_size=config.batch_size,
                        learning_rate=config.learning_rate, update_frequent=config.update_frequent, debug_log_frequent=config.debug_log_frequent,
                        gamma=config.gamma, tau=config.tau, K = config.K, log_dir=config.log_dir, actor_network=config.actor_network,
                        critic_network=config.critic_network, actor_hidden_dim=config.actor_hidden_dim, model_hidden_dim=config.model_hidden_dim,
                        critic_hidden_dim=config.critic_hidden_dim, n_steps_train=config.n_steps_train,
                        env_name=envName, init_train_steps=config.init_train_steps,
                        network_size=config.network_size, elite_size=config.elite_size, use_decay=config.use_decay,
                        model_batch_size=config.model_batch_size, model_train_freq=config.model_train_freq, n_steps_model=config.n_steps_model,
                        rollout_length_range=config.rollout_length_range, rollout_epoch_range=config.rollout_epoch_range,
                        rollout_batch_size=config.rollout_batch_size, real_ratio=config.real_ratio,
                        model_retain_epochs=config.model_retain_epochs, batch_size_d=config.batch_size_d)#gamma=0.95
    save_parameter_setting(agent.log_dir,"g_matd3",config)
    if config.use_init_bc and not debug:
        init_offline_train(agent, "./utils/data/exp/{}/experience.pkl".format(offline_data), config.init_bc_steps, lock=lock)
    data = agent.learning(
                  decaying_epsilon=True,
                  epsilon_high=1.0,  #if not config.use_init_bc else 0.1
                  epsilon_low=0.01 , #if not config.use_init_bc else 0.1
                  max_episode_num=config.max_episode,
                  explore_episodes_percent=1.0,
                  init_exp_file="./utils/data/exp/{}/experience.pkl".format(random_data),
                  lock=lock
                 )
    for i in range(agent.env.agent_count):
        agent.save(agent.log_dir,"Actor{}".format(i),agent.agents[i].actor, config.max_episode)

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
        agent.save(agent.log_dir,"Actor{}".format(i),agent.agents[i].actor, config.max_episode)
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
        agent.save(agent.log_dir,"Actor{}".format(i),agent.agents[i].actor, config.max_episode)
    learning_curve(data, 2, 1, step_index=0, title="MADDPGAgent performance on {}".format(envName),
                   x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName="maddpg")

def run_experiment_once(args, id, lock=None):
    env_dict = {
        "map_02": map_02,
        "map_05": map_05,
        "map_06": map_06,
        "map_10": map_10,
        # "map_11": map_11,
        # "map_12": map_12
    }
    envName, env = ("PedsMoveEnv",my_env.PedsMoveEnv(terrain=env_dict[args.map], person_num=args.p_num,
                                                     group_size=(args.g_size, args.g_size), maxStep=args.max_step,
                                                     random_init_mode=args.random_init))

    config = DebugConfig()
    #config = PedsMoveConfig(n_rol_threads=args.threads, max_episode=args.train_step)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    if args.dir != './':
        os.mkdir("./" + args.dir + "/" + str(id))
    config.log_dir = args.dir + "/" + str(id)

    config.use_init_bc = True
    print("当前测试环境:{}".format(env.terrain.name))
    config.n_steps_train = 10
    print("GD-MAMBPO gradient descent training!")
    test2(env, envName, config=config, lock=lock)  # Model-Based Matd3 10Step(GD-MAMBPO)
    # print("MATD3-10 gradient descent training!")
    # test1(env, envName, config=config)  # Matd3 10Step No BC(MATD3-10)


    config.n_steps_train = 10  # 1
    config.use_init_bc = False
    print("MAMBPO gradient descent training!")
    test2(env, envName, config=config)  # Model-Based Matd3 10Step No BC(MAMBPO)
    print("MATD3-10 gradient descent training!")
    test1(env, envName, config=config)  # Matd3 10Step No BC(MATD3-10)

# xvfb-run -a python run.py --dir train_05 --map map_05 --train_step 400  --max_step 250 --threads 2 --p_num=8 --g_size=1
# xvfb-run -a python run.py --dir train_05 --map map_05 --train_step 400  --max_step 250 --threads 2
# xvfb-run -a python run.py --dir train_06 --map map_06 --train_step 300  --max_step 250 --threads 2
# xvfb-run -a python run.py --dir train_02 --map map_02 --train_step 300  --max_step 250 --threads 2
# xvfb-run -a python run.py --dir train_map_01 --map map_01 --max_step 2000 --train_step 500 --threads 2

# xvfb-run -a python run.py --dir train_10 --map map_10 --train_step 400  --max_step 250 --threads 2 --p_num=32 --g_size=4
# xvfb-run -a python run.py --dir train_11 --map map_11 --train_step 400  --max_step 250 --threads 2 --p_num=32 --g_size=4
# xvfb-run -a python run.py --dir train_12 --map map_12 --train_step 400  --max_step 250 --threads 2 --p_num=32 --g_size=4

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description="Run PedsMoveEnv use reinforcement learning algroithms!")
    my_parser.add_argument('--dir', default="01_train", type=str)
    my_parser.add_argument('--map', default="map_05", type=str)
    my_parser.add_argument('--p_num', default=30, type=int)
    my_parser.add_argument('--g_size', default=5, type=int)
    my_parser.add_argument('--max_step', default=250, type=int)
    my_parser.add_argument('--count', default=2, type=int)
    my_parser.add_argument('--train_step', default=100, type=int)
    my_parser.add_argument('--random_init', default=True, type=bool)
    my_parser.add_argument('--threads', default=2, type=int)
    args = my_parser.parse_args()

    if args.dir != './':
        os.mkdir("./" + args.dir)
    if args.count == 1:#采用单线程的形式
        run_experiment_once(args, "1")
    else:
        lock = multiprocessing.Lock()
        sub_processes = []
        for i in range(args.count):
            p = Process(target=run_experiment_once, args=(args, i, lock))
            p.start()
            sub_processes.append(p)
        for p in sub_processes:
            p.join()
        print("主进程结束!")

