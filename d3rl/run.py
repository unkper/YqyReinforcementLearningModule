import copy

import d3rlpy
import numpy as np
import pyglet

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
            'value_scale': d3rlpy.metrics.average_value_estimation_scorer
        },
    )

    cql.save_model("model")


from d3rlpy_multiEnv_extend import evaluate_on_muit_environment, train_mult_env
from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import map_simple
import d3rlpy_multiEnv_extend as ex
from ped_env.mdp import PedsRLHandler


def test4():
    env = PedsMoveEnv(map_simple, group_size=(1, 1), person_handler=PedsRLHandler)

    scorer = evaluate_on_muit_environment(env, 10, 3, render=True)

    dqn = d3rlpy.algos.DQN()
    dqn.build_with_env(env)
    dqn.load_model(
        r"D:\projects\python\PedestrainSimulationModule\d3rl\d3rlpy_logs\DQN_online_20221209015145\model_100000.pt")

    print("Mean value is:{}".format(scorer(dqn)))


def test5():
    env = PedsMoveEnv(map_simple, group_size=(1, 1), person_handler=PedsRLHandler)

    eval_env = copy.deepcopy(env)

    dqn = d3rlpy.algos.DiscreteSAC(use_gpu=True)
    dqn.build_with_env(env)

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
        n_steps_per_epoch=4000,  # evaluation is performed every 1K steps
        update_start_step=1000,  # parameter update starts after 1K steps
        tensorboard_dir="runs"
    )

def test6():
    window = pyglet.window.Window()
    label = pyglet.text.Label('Hello, world',
                              font_name='Times New Roman',
                              font_size=36,
                              x=window.width // 2, y=window.height // 2,
                              anchor_x='center', anchor_y='center')

    @window.event
    def on_draw():
        window.clear()
        label.draw()

    pyglet.app.run()


if __name__ == '__main__':
    test6()
