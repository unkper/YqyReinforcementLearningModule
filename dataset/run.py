import d3rlpy
import numpy as np

from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import map_11
from dataset.data_recorder import load_dataset


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
        },
    )

    cql.save_model("model")

if __name__ == '__main__':
    test2()
