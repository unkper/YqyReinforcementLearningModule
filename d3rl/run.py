import copy

import d3rlpy
import numpy as np

from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import map_11
from d3rl.data_recorder import load_dataset


def test1():
    env = PedsMoveEnv(map_11, 40, (1, 1), discrete=False)
    from random_policy import RandomPolicy
    from data_recorder import DataRecorder
    policy = RandomPolicy(np.array([2]))
    recorder = DataRecorder(env, policy, 40)
    recorder.save()


def test3():
    env = PedsMoveEnv(map_11, 40, (1, 1), discrete=False)
    from random_policy import RandomPolicy
    from data_recorder import DataRecorder, MultiProcessRecorder
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
            'value_scale':d3rlpy.metrics.average_value_estimation_scorer
        },
    )

    cql.save_model("model")

from train_muit_env import evaluate_on_muit_environment, train_mult_env
from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import map_simple

def test4():
    env = PedsMoveEnv(map_simple, group_size=(1, 1))

    scorer = evaluate_on_muit_environment(env, 10, 3, render=True)

    dqn = d3rlpy.algos.DQN()
    dqn.build_with_muit_env(env)

    print("Mean value is:{}".format(scorer(dqn)))

def test5():
    env = PedsMoveEnv(map_simple, group_size=(1, 1))

    eval_env = copy.deepcopy(env)

    dqn = d3rlpy.algos.DQN()
    dqn.build_with_muit_env(env)

    # experience replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

    # exploration strategy
    explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy()

    train_mult_env(
        dqn,
        10,
        env,
        buffer,
        explorer,
        save_interval=10,
        n_steps=100000,  # train for 100K steps
        eval_env=eval_env,
        n_steps_per_epoch=1000,  # evaluation is performed every 1K steps
        update_start_step=1000,  # parameter update starts after 1K steps
        tensorboard_dir="runs"
    )


if __name__ == '__main__':
    test5()
