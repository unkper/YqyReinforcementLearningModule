import d3rlpy
from sklearn.model_selection import train_test_split


def test():
    dataset, env = d3rlpy.datasets.get_atari('breakout-expert-v0')

    alg = d3rlpy.algos.DiscreteSAC(use_gpu=True)
    alg.build_with_dataset(dataset)
    alg.load_model(
        r'D:\projects\python\PedestrainSimulationModule\dataset\d3rlpy_logs\breakout_20221130210258\model_100000.pt')

    from d3rlpy.metrics.scorer import evaluate_on_environment

    scorer = evaluate_on_environment(env, render=True)
    scorer(alg)


def train():
    # prepare rl_platform
    dataset, env = d3rlpy.datasets.get_atari('breakout-expert-v0')

    alg = d3rlpy.algos.DiscreteSAC(use_gpu=True)

    metrics = {"td_error": d3rlpy.metrics.scorer.td_error_scorer,
               "exp": d3rlpy.metrics.scorer.evaluate_on_environment(env)}

    alg.fit(dataset,
            n_steps_per_epoch=1000,
            n_steps=100000,
            experiment_name="breakout",
            save_interval=10,
            scorers=metrics,
            tensorboard_dir="runs")

    print("hello")


test()
