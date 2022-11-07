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
from rl.utils.planners import load_offline_train
from rl.agents.Matd3Agent import MATD3Agent
from rl.agents.MAMBPOAgent import MAMBPOAgent
from rl.utils.miscellaneous import save_experiment_data, save_parameter_setting

from rl.config import *


def test1(useEnv, envName, config: Config, lock=None):
    alg_name = "MATD3"
    config.update_parameter(alg_name)
    offline_data = useEnv.terrain.name + "_exp"
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    d_exp = None
    if config.use_init_bc:
        d_exp = load_offline_train(useEnv, "./utils/data/exp/{}/experience.pkl".format(offline_data), lock=lock)
    agent = MATD3Agent(env, n_rol_threads=config.n_rol_threads, capacity=config.capacity, batch_size=config.batch_size,
                       learning_rate=config.learning_rate, update_frequent=config.update_frequent,
                       debug_log_frequent=config.debug_log_frequent,
                       gamma=config.gamma, tau=config.tau, K=config.K, log_dir=config.log_dir,
                       actor_network=config.actor_network,
                       critic_network=config.critic_network, actor_hidden_dim=config.actor_hidden_dim,
                       critic_hidden_dim=config.critic_hidden_dim, n_steps_train=config.n_steps_train,
                       env_name=envName, demo_experience=d_exp, batch_size_d=config.batch_size_d,
                       lambda_1=config.lambda1, lambda_2=config.lambda2)
    save_parameter_setting(agent.log_dir, alg_name, config)
    data = agent.learning(
        decaying_epsilon=config.decaying_epsilon,
        epsilon_high=config.epsilon_high,
        epsilon_low=config.epsilon_low,
        max_episode_num=config.max_episode,
        explore_episodes_percent=0.8,
    )
    for i in range(agent.env.agent_count):
        agent.save(agent.log_dir, "Actor{}".format(i), agent.agents[i].actor, config.max_episode)
    save_experiment_data(data, 2, 1, step_index=0, title="{}Agent performance on {}".format(alg_name, envName),
                         x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName=alg_name)


def test2(useEnv, envName, config: Config, debug=False, lock=None):
    alg_name = "GD_MAMBPO" if config.use_init_bc else "MAMBPO"
    if not debug:
        offline_data = useEnv.terrain.name + "_exp"
        random_data = useEnv.terrain.name + "_exp_random"
    config.update_parameter(alg_name)
    env = useEnv
    print(env.observation_space)
    print(env.action_space)
    d_exp = None
    if config.use_init_bc:
        d_exp = load_offline_train(useEnv, "./utils/data/exp/{}/experience.pkl".format(offline_data), lock=lock)
    agent = MAMBPOAgent(env, n_rol_threads=config.n_rol_threads, capacity=config.capacity, batch_size=config.batch_size,
                        learning_rate=config.learning_rate, update_frequent=config.update_frequent,
                        debug_log_frequent=config.debug_log_frequent,
                        gamma=config.gamma, tau=config.tau, K=config.K, log_dir=config.log_dir,
                        actor_network=config.actor_network,
                        critic_network=config.critic_network, actor_hidden_dim=config.actor_hidden_dim,
                        model_hidden_dim=config.model_hidden_dim,
                        critic_hidden_dim=config.critic_hidden_dim, n_steps_train=config.n_steps_train,
                        env_name=envName, init_train_steps=config.init_train_steps,
                        network_size=config.network_size, elite_size=config.elite_size, use_decay=config.use_decay,
                        model_batch_size=config.model_batch_size, model_train_freq=config.model_train_freq,
                        n_steps_model=config.n_steps_model,
                        rollout_length_range=config.rollout_length_range,
                        rollout_epoch_range=config.rollout_epoch_range,
                        rollout_batch_size=config.rollout_batch_size, real_ratio=config.real_ratio,
                        model_retain_epochs=config.model_retain_epochs, demo_experience=d_exp,
                        batch_size_d=config.batch_size_d,
                        lambda_1=config.lambda1, lambda_2=config.lambda2)
    save_parameter_setting(agent.log_dir, alg_name, config)
    data = agent.learning(
        decaying_epsilon=config.decaying_epsilon,
        epsilon_high=config.epsilon_high,
        epsilon_low=config.epsilon_low,
        max_episode_num=config.max_episode,
        explore_episodes_percent=0.8,
        init_exp_file="./utils/data/exp/{}/experience.pkl".format(random_data),
        lock=lock
    )
    for i in range(agent.env.agent_count):
        agent.save(agent.log_dir, "Actor{}".format(i), agent.agents[i].actor, config.max_episode)
    save_experiment_data(data, 2, 1, step_index=0, title="{}Agent performance on {}".format(alg_name, envName),
                         x_name="episodes", y_name="rewards of episode", save_dir=agent.log_dir, saveName=alg_name)


def run_experiment_once(args, id, lock=None):
    env_dict = {
        "map_02": map_02,
        "map_05": map_05,
        "map_06": map_06,
        "map_10": map_10,
        "map_11": map_11,
        "map_12": map_12
    }
    envName, env = ("PedsMoveEnv", my_env.PedsMoveEnv(terrain=env_dict[args.map], person_num=args.p_num,
                                                      group_size=(args.g_size, args.g_size), maxStep=args.max_step,
                                                      random_init_mode=args.random_init,
                                                      frame_skipping=args.frame_skip))

    # config = DebugConfig()

    config = PedsMoveConfig(n_rol_threads=args.threads, max_episode=args.train_step, use_decay_epsilon=args.use_decay)

    if args.dir != './':
        os.mkdir("./" + args.dir + "/" + str(id))
    config.log_dir = args.dir + "/" + str(id)

    config.use_init_bc = True
    print("当前测试环境:{}".format(env.terrain.name))
    config.n_steps_train = 10
    # print("GD-MAMBPO gradient descent training!")
    # test2(env, envName, config=config, lock=lock)  # Model-Based Matd3 10Step(GD-MAMBPO)

    # print("BC-MATD3-10 gradient descent training!")
    # test1(env, envName, config=config)  # Matd3 10Step No BC(MATD3-10)

    config.n_steps_train = 10  # 1
    config.use_init_bc = False
    print("MAMBPO gradient descent training!")
    test2(env, envName, config=config)  # Model-Based Matd3 10Step No BC(MAMBPO)
    #
    print("MATD3-10 gradient descent training!")
    test1(env, envName, config=config)  # Matd3 10Step No BC(MATD3-10)

    # config.n_steps_train = 1
    # test1(env, envName, config=config)  # Matd3 1Step No BC(MATD3-1)


# xvfb-run -a python run.py --dir train_05 --map map_05 --train_step 400  --max_step 250 --threads 2 --p_num=8 --g_size=1
# xvfb-run -a python run.py --dir train_05 --map map_05 --train_step 400  --max_step 250 --threads 2
# xvfb-run -a python run.py --dir train_06 --map map_06 --train_step 300  --max_step 250 --threads 2
# xvfb-run -a python run.py --dir train_02 --map map_02 --train_step 300  --max_step 250 --threads 2
# xvfb-run -a python run.py --dir train_map_01 --map map_01 --max_step 2000 --train_step 500 --threads 2

# xvfb-run -a python run.py --dir train_10 --map map_10 --train_step 400  --max_step 250 --threads 2 --p_num=20 --g_size=1 --count=1
# xvfb-run -a python run.py --dir train_11 --map map_11 --train_step 400  --max_step 750 --threads 2 --p_num=40 --g_size=5 --count=5
# xvfb-run -a python run.py --dir train_12 --map map_12 --train_step 600  --max_step 250 --threads 2 --p_num=32 --g_size=4 --count=5

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description="Run PedsMoveEnv use reinforcement learning algroithms!")
    my_parser.add_argument('--dir', default="01_train", type=str)
    my_parser.add_argument('--map', default="map_11", type=str)
    my_parser.add_argument('--p_num', default=32, type=int)
    my_parser.add_argument('--g_size', default=4, type=int)
    my_parser.add_argument('--max_step', default=500, type=int)
    my_parser.add_argument('--count', default=1, type=int)
    my_parser.add_argument('--frame_skip', default=8, type=int)
    my_parser.add_argument('--train_step', default=20, type=int)
    my_parser.add_argument('--random_init', default=True, type=bool)
    my_parser.add_argument('--threads', default=2, type=int)
    my_parser.add_argument('--use_decay', default=False, type=bool)
    args = my_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if args.dir != './':
        os.mkdir("./" + args.dir)
    if args.count == 1:  # 采用单线程的形式
        run_experiment_once(args, "1")
    else:
        torch.multiprocessing.set_start_method("spawn")
        lock = multiprocessing.Lock()
        sub_processes = []
        for i in range(args.count):
            p = Process(target=run_experiment_once, args=(args, i, lock))
            p.start()
            sub_processes.append(p)
        for p in sub_processes:
            p.join()
        print("主进程结束!")
