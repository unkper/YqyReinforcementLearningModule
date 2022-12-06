import d3rlpy.algos


def test_dqn():
    import gym

    # for training
    env = gym.make("CartPole-v1")

    eval_env = gym.make("CartPole-v1")

    alg = d3rlpy.algos.DQN(use_gpu=True)

    # experience replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

    # exploration strategy
    # in this tutorial, epsilon-greedy policy with static epsilon=0.3
    explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy()
    alg.build_with_env(env)
    alg.fit_online(
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
    test_dqn()