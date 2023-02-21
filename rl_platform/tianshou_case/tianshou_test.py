import copy
import datetime
import os

import gym
import pettingzoo as pet
import tianshou as ts
import torch
from tensorboardX import SummaryWriter
from tianshou.policy import BasePolicy

from ped_env.envs import PedsMoveEnv
from ped_env.utils.maps import map_08, map_10, map_simple
from rl_platform.tianshou_case.utils.wrapper import FrameStackWrapper


def lunarlander_test(model_path):
    task = "LunarLander-v2"
    from tianshou.utils.net.common import Net
    env = gym.make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    eps_test = 0.05
    gamma, n_step, target_freq = 0.9, 3, 320
    lr = 1e-3

    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)

    policy.load_state_dict(torch.load(model_path))

    policy.eval()
    policy.set_eps(eps_test)
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=1, render=1 / 20)


def lunarlander_train(save_path=None):
    task = "LunarLander-v2"
    file_name = task + "_DQN_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    device = torch.device("cuda:0")

    lr, epoch, batch_size = 1e-3, 200, 256
    train_num, test_num = 10, 10
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 50000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 10000, 10
    logger = ts.utils.TensorboardLogger(SummaryWriter('log/' + file_name))  # TensorBoard is supported!

    # you can also try with SubprocVectorEnv
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

    from tianshou.utils.net.common import Net
    # you can define other net by following the API:
    # https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network
    env = gym.make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128],
              device=device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs,
                                       exploration_noise=True)  # because DQN uses epsilon-greedy method

    if save_path is not None:
        # load from existing checkpoint
        print(f"Loading agent under {save_path}")
        ckpt_path = os.path.join(save_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cuda:0"))
            policy.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html

        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        path = 'log/{}/model_{}.pth'.format(file_name, epoch)

        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            }, path
        )
        return path

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
        test_num, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        save_best_fn=lambda policy: torch.save(policy.state_dict(), 'log/{}/best_model.pth'.format(file_name)),
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger)
    print(f'Finished training! Use {result["duration"]}')

    torch.save(policy.state_dict(), 'log/{}/dqn_{}.pth'.format(file_name,
                                                               datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

from pettingzoo.classic import rps_v2, tictactoe_v3


def multiagent_train1():
    from pettingzoo.mpe import simple_world_comm_v2

    if __name__ == "__main__":
        # Step 1: Load the PettingZoo environment
        env = pet.utils.parallel_to_aec(simple_world_comm_v2.parallel_env())
        obs = env.reset()
        is_done = False
        while not is_done:
            actions = {agent: env.action_space(agent).sample() for agent in
                       env.agents}  # this is where you would insert your policy
            observations, rewards, terminations, truncations, infos = env.step(actions)
            is_done = all(terminations)

        # # Step 3: Define policies for each agent
        # policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)
        #
        # # Step 4: Convert the env to vector format
        # env = DummyVectorEnv([lambda: env])
        #
        # # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
        # collector = Collector(policies, env)
        #
        # # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
        # result = collector.collect(n_episode=1, render=0.1)

def multiagent_train2():
    """This is a minimal example of using Tianshou with MARL to train agents.

    Author: Will (https://github.com/WillDudley)

    Python version used: 3.8.10

    Requirements:
    pettingzoo == 1.22.0
    git+https://github.com/thu-ml/tianshou
    """

    import os
    from typing import Optional, Tuple

    import gym
    import numpy as np
    import torch
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.env.pettingzoo_env import PettingZooEnv
    from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
    from tianshou.trainer import offpolicy_trainer
    from tianshou.utils.net.common import Net

    from pettingzoo.classic import tictactoe_v3

    def _get_agents(
            agent_learn: Optional[BasePolicy] = None,
            agent_count: int=1,
            optim: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
        env = _get_env()
        observation_space = (
            env.observation_space["observation"]
            if isinstance(env.observation_space, gym.spaces.Dict)
            else env.observation_space
        )
        if agent_learn is None:
            # model
            net = Net(
                state_shape=observation_space.shape
                            or observation_space.n,
                action_shape=env.action_space.shape or env.action_space.n,
                hidden_sizes=[128, 128, 128, 128],
                device="cuda" if torch.cuda.is_available() else "cpu",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            if optim is None:
                optim = torch.optim.Adam(net.parameters(), lr=1e-4)
            agent_learn = DQNPolicy(
                model=net,
                optim=optim,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=320,
            )

        agents = [copy.deepcopy(agent_learn) for _ in range(agent_count)]
        policy = MultiAgentPolicyManager(agents, env)
        return policy, optim, env.agents

    agent_num_map8 = 8
    def _get_env():
        """This function is needed to provide callables for DummyVectorEnv."""
        temp = pet.utils.parallel_to_aec(
            PedsMoveEnv(map_08, person_num=agent_num_map8, group_size=(1, 1), random_init_mode=True))
        return PettingZooEnv(temp)

    if __name__ == "__main__":
        # ======== Step 1: Environment setup =========
        train_envs = DummyVectorEnv([_get_env for _ in range(10)])
        test_envs = DummyVectorEnv([_get_env for _ in range(10)])

        # seed
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)

        # ======== Step 2: Agent setup =========
        policy, optim, agents = _get_agents(agent_count=agent_num_map8)

        # ======== Step 3: Collector setup =========
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(100000, len(train_envs)),
            exploration_noise=True,
        )
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=64 * 10)  # batch size * training_num

        # ======== Step 4: Callback functions setup =========
        task = "PedsMoveEnv"
        file_name = task + "_DQN_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logger = ts.utils.TensorboardLogger(SummaryWriter('log/rps/dqn' + file_name))  # TensorBoard is supported!
        def save_best_fn(policy):
            model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
            os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
            torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

        def stop_fn(mean_rewards):
            return mean_rewards >= 0.6

        def train_fn(epoch, env_step):
            policy.policies[agents[1]].set_eps(0.1)

        def test_fn(epoch, env_step):
            policy.policies[agents[1]].set_eps(0.05)

        def reward_metric(rews):
            return rews[:, 1]

        # ======== Step 5: Run the trainer =========
        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=50,
            step_per_epoch=1000,
            step_per_collect=50,
            episode_per_test=10,
            batch_size=64,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            update_per_step=0.1,
            test_in_train=False,
            reward_metric=reward_metric,
            logger = logger
        )

        # return result, policy.policies[agents[1]]
        print(f"\n==========Result==========\n{result}")
        print("\n(the trained policy can be accessed via policy.policies[agents[1]])")


if __name__ == '__main__':

    from pettingzoo.test import parallel_api_test, api_test

    env = FrameStackWrapper(PedsMoveEnv(map_simple, person_num=6, group_size=(1, 1), random_init_mode=True, maxStep=10000))
    for _ in range(100):
        parallel_api_test(env, num_cycles=4000)

    #multiagent_train1()
    #multiagent_train2()
    #lunarlander_train()
    #lunarlander_test(
    #  r"D:\projects\python\PedestrainSimulationModule\rl_platform\log\LunarLander-v2_DQN_2022_12_14_21_48_56\best_model.pth")
