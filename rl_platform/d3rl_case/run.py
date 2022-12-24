import copy

import d3rlpy
import numpy as np

from ped_env.utils.maps import map_11, map_08
from d3rl.d3rl_case.data_recorder import load_dataset


def test1():
    env = PedsMoveEnv(map_11, 40, (1, 1), discrete=False)
    from d3rl.d3rl_case.random_policy import RandomPolicy
    from d3rl.d3rl_case.data_recorder import DataRecorder
    policy = RandomPolicy(np.array([2]))
    recorder = DataRecorder(env, policy, 40)
    recorder.save()


def test3():
    env = PedsMoveEnv(map_11, 40, (1, 1), discrete=False)
    from d3rl.d3rl_case.random_policy import RandomPolicy
    from d3rl.d3rl_case.data_recorder import DataRecorder, MultiProcessRecorder
    policy = RandomPolicy(np.array([2]))
    recorder = DataRecorder(env, policy, 40)
    mult = MultiProcessRecorder(recorder, 5)
    dataset = mult.collect(1)
    mult.save(dataset)


def test2():
    dataset = load_dataset("./datasets/dataset_2022_11_08_21_09_49.pkl")

    # prepare algorithm
    cql = d3rlpy.algos.CQL(use_gpu=False)

    # train
    cql.fit(
        dataset,
        n_epochs=100,
        scorers={
            'td_error': d3rlpy.metrics.td_error_scorer,
            'value_scale': d3rlpy.metrics.average_value_estimation_scorer
        },
    )

    cql.save_model("model")


from d3rl.d3rl_case.d3rlpy_extend import evaluate_on_muit_environment, train_mult_env
from ped_env.envs import PedsMoveEnv


def evaluation(env: PedsMoveEnv, algo: d3rlpy.algos.AlgoBase, model_path, ep=3):
    scorer = evaluate_on_muit_environment(env, env.agent_count, ep, render=True)

    algo.build_with_env(env)
    algo.load_model(model_path)

    print("Mean value is:{}".format(scorer(algo)))


def train(env, algo:d3rlpy.algos.AlgoBase, agent_num, model_name: str = None):
    eval_env = copy.deepcopy(env)

    algo.build_with_env(env)
    if model_name is not None:
        algo.load_model(model_name)

    # experience replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

    # exploration strategy
    explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy()

    train_mult_env(
        algo,
        agent_num,
        env,
        buffer,
        explorer,
        save_interval=10,
        n_steps=1000000,  # train for 100K steps
        eval_env=eval_env,
        n_steps_per_epoch=4000,  # evaluation is performed every 1K steps
        update_start_step=1000,  # parameter update starts after 1K steps
        tensorboard_dir="runs"
    )


if __name__ == '__main__':
    # agent_num = 20
    agent_num_map8 = 8

    env = PedsMoveEnv(map_08, person_num=agent_num_map8, group_size=(1, 1), random_init_mode=True)
    algo = d3rlpy.algos.DoubleDQN(use_gpu=True)


    #train(env, algo, agent_num_map8, model_name=r"D:\projects\python\PedestrainSimulationModule\rl_platform\d3rlpy_logs\DoubleDQN_online_20221213003720\model_1000000.pt")

    evaluation(env, algo, r"/rl_platform\d3rlpy_logs\SAC_online_20221213191403\model_1000000.pt")
