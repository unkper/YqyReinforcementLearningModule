from typing import Optional, Callable, Any, cast, Sequence

import numpy as np
from d3rlpy.logger import D3RLPyLogger
from d3rlpy.metrics import evaluate_on_environment
from d3rlpy.online.buffers import Buffer
from d3rlpy.online.explorers import Explorer
from d3rlpy.online.iterators import AlgoProtocol, _setup_algo
import gym
from d3rlpy.preprocessing.stack import StackedObservation
from gym.spaces import Discrete
from tqdm import trange


def onehot_from_int(x: int, action_dim: int = 9) -> np.ndarray:
    return np.array([0 if i != x else 1 for i in range(action_dim)])


def onehot_to_int(x, action_dim: int = 9):
    for i in range(0, action_dim):
        if i == 1:
            return i


def get_action_size_from_env(env: gym.Env) -> int:
    if isinstance(env.action_space[0], Discrete):
        return cast(int, env.action_space[0].n)
    return cast(int, env.action_space[0].shape[0])


def build_with_muit_env(alg, env: gym.Env) -> None:
    """Instantiate implementation object with OpenAI Gym object.

    Args:
        env: gym-like environment.

    """
    observation_shape = env.observation_space[0].shape
    alg.create_impl(
        alg._process_observation_shape(observation_shape),
        get_action_size_from_env(env),
    )


def evaluate_on_muit_environment(
        env: gym.Env, agent_num: int, n_trials: int = 10, epsilon: float = 0.0, render: bool = False
) -> Callable[..., float]:
    """Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.

    Returns:
        scoerer function.


    """

    # for image observation

    def scorer(algo: AlgoProtocol, *args: Any) -> float:

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            while True:
                # take action
                actions = []
                for i in range(agent_num):
                    if np.random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        action = algo.predict([np.array(observation[i])])[0]
                    actions.append(onehot_from_int(action))

                observation, reward, done, _ = env.step(actions)
                episode_reward += np.mean(reward)

                if render:
                    env.render()

                if all(done):
                    break
            episode_rewards.append(episode_reward)
        return float(np.mean(episode_rewards))

    return scorer


def train_mult_env(
        algo: AlgoProtocol,
        agent_num: int,
        env: gym.Env,
        buffer: Buffer,
        explorer: Optional[Explorer] = None,
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        update_interval: int = 1,
        update_start_step: int = 0,
        random_steps: int = 0,
        eval_env: Optional[gym.Env] = None,
        eval_epsilon: float = 0.0,
        save_metrics: bool = True,
        save_interval: int = 1,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        timelimit_aware: bool = True,
        callback: Optional[Callable[[AlgoProtocol, int, int], None]] = None,
) -> None:
    """Start training loop of online deep reinforcement learning.

    Args:
        algo: algorithm object.
        env: gym-like environment.
        buffer : replay buffer.
        explorer: action explorer.
        n_steps: the number of total steps to train.
        n_steps_per_epoch: the number of steps per epoch.
        update_interval: the number of steps per update.
        update_start_step: the steps before starting updates.
        random_steps: the steps for the initial random explortion.
        eval_env: gym-like environment. If None, evaluation is skipped.
        eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
        save_metrics: flag to record metrics. If False, the log
            directory is not created and the model parameters are not saved.
        save_interval: the number of epochs before saving models.
        experiment_name: experiment name for logging. If not passed,
            the directory name will be ``{class name}_online_{timestamp}``.
        with_timestamp: flag to add timestamp string to the last of
            directory name.
        logdir: root directory name to save logs.
        verbose: flag to show logged information on stdout.
        show_progress: flag to show progress bar for iterations.
        tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
        timelimit_aware: flag to turn ``terminal`` flag ``False`` when
            ``TimeLimit.truncated`` flag is ``True``, which is designed to
            incorporate with ``gym.wrappers.TimeLimit``.
        callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called at the end of epochs.

    """
    # setup logger
    if experiment_name is None:
        experiment_name = algo.__class__.__name__ + "_online"

    logger = D3RLPyLogger(
        experiment_name,
        save_metrics=save_metrics,
        root_dir=logdir,
        verbose=verbose,
        tensorboard_dir=tensorboard_dir,
        with_timestamp=with_timestamp,
    )
    algo.set_active_logger(logger)

    # initialize algorithm parameters
    _setup_algo(algo, env)

    # save hyperparameters
    algo.save_params(logger)

    # switch based on show_progress flag
    xrange = trange if show_progress else range

    # setup evaluation scorer
    eval_scorer: Optional[Callable[..., float]]
    if eval_env:
        eval_scorer = evaluate_on_muit_environment(eval_env, agent_num, n_trials=5, epsilon=eval_epsilon)
    else:
        eval_scorer = None

    # start training loop
    observation = env.reset()
    rollout_return = 0.0
    for total_step in xrange(1, n_steps + 1):
        with logger.measure_time("step"):
            # stack observation if necessary
            new_observation = []
            for ob in observation:
                ob = np.array(ob).astype("f4")
                new_observation.append(ob)
            fed_observation = new_observation

            actions = []
            # sample exploration action
            with logger.measure_time("inference"):
                for i in range(agent_num):
                    if total_step < random_steps:
                        action = env.action_space.sample()
                    elif explorer:
                        x = fed_observation[i].reshape((1,) + fed_observation[i].shape)
                        action = explorer.sample(algo, x, total_step)[0]
                    else:
                        action = algo.sample_action([fed_observation[i]])[0]
                    actions.append(onehot_from_int(action))

            # step environment
            with logger.measure_time("environment_step"):
                next_observation, reward, terminal, info = env.step(actions)
                rollout_return += sum(reward)

            # special case for TimeLimit wrapper
            if timelimit_aware and "TimeLimit.truncated" in info:
                clip_episode = True
                terminal = [False for _ in range(agent_num)]
            else:
                clip_episode = any(terminal)  # 所有智能体都结束了才结束

            for i in range(agent_num):
                # store observation
                buffer.append(
                    observation=np.array(observation[i]),
                    action=onehot_to_int(actions[i]),
                    reward=reward[i],
                    terminal=terminal[i],
                    clip_episode=clip_episode,
                )

            # reset if terminated
            if clip_episode:
                observation = env.reset()
                logger.add_metric("rollout_return", rollout_return)
                rollout_return = 0.0
            else:
                observation = next_observation

            # psuedo epoch count
            epoch = total_step // n_steps_per_epoch

            if total_step > update_start_step and len(buffer) > algo.batch_size:
                if total_step % update_interval == 0:
                    # sample mini-batch
                    with logger.measure_time("sample_batch"):
                        batch = buffer.sample(
                            batch_size=algo.batch_size,
                            n_frames=algo.n_frames,
                            n_steps=algo.n_steps,
                            gamma=algo.gamma,
                        )

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = algo.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)

            # call callback if given
            if callback:
                callback(algo, epoch, total_step)

        if epoch > 0 and total_step % n_steps_per_epoch == 0:
            # evaluation
            if eval_scorer:
                logger.add_metric("evaluation", eval_scorer(algo))

            if epoch % save_interval == 0:
                logger.save_model(total_step, algo)

            # save metrics
            logger.commit(epoch, total_step)

    # clip the last episode
    buffer.clip_episode()

    # close logger
    logger.close()
