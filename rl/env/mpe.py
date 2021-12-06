import gym
import numpy as np

from gym.spaces import Box, Discrete
from numpy import inf

from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_speaker_listener_v3
from pettingzoo.mpe import simple_push_v2
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_world_comm_v2
from pettingzoo.mpe import simple_crypto_v2
from pettingzoo.mpe import simple_reference_v2
from pettingzoo.mpe import simple_tag_v2

class GymEnvWrapper(gym.Env):
    def __init__(self):
        super(GymEnvWrapper, self).__init__()
        self.agent_count = -1
        self.owner = False

    def reset(self):
        obs = self.wrappedEnv.reset()
        if not isinstance(obs, list):
            obs = list(obs.values())
        return obs

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

    def close(self):
        self.wrappedEnv.close()

    def map_value(self, data, MIN=0, MAX=1):
        """
        归一化映射到任意区间
        :param data: 数据
        :param MIN: 目标数据最小值
        :param MAX: 目标数据最小值
        :return:
        """
        d_min = -1  # 当前数据最大值
        d_max = 1  # 当前数据最小值
        return MIN + (MAX - MIN) / (d_max - d_min) * (data - d_min)

    def step(self, action):
        if self.discrete:
            _action = [np.argmax(x).item() for x in action]
        else:
            _action = self.map_value(action).astype(np.float32)
        if not self.owner:
            actions = {agent: _action[idx] for idx, agent in enumerate(self.wrappedEnv.agents)}
        else:
            actions = _action
        obs, reward, is_done, info = self.wrappedEnv.step(actions)
        if not isinstance(obs, list):
            obs = list(obs.values())
            reward = list(reward.values())
            is_done = list(is_done.values())
        return obs, reward, is_done, "MPE"

class SimpleAdversary(GymEnvWrapper):
    def __init__(self, maxStep = 25):
        super(SimpleAdversary, self).__init__()
        self.wrappedEnv = simple_adversary_v2.parallel_env(N=2, max_cycles = maxStep, continuous_actions=False)
        self.agent_count = 3
        self.observation_space = [Box(-inf, inf, (8,)), Box(-inf, inf, (10,)), Box(-inf, inf, (10,))]
        self.action_space = [Discrete(5),Discrete(5),Discrete(5)]

class SimpleSpeakerListener(GymEnvWrapper):
    def __init__(self,maxStep=25):
        super(SimpleSpeakerListener, self).__init__()
        self.wrappedEnv = simple_speaker_listener_v3.parallel_env(max_cycles=maxStep, continuous_actions=False)
        self.agent_count = 2
        self.observation_space = [Box(-inf, inf, (3,)), Box(-inf, inf, (11,))]
        self.action_space = [Discrete(3),Discrete(5)]

class SimplePusher(GymEnvWrapper):
    def __init__(self,maxStep=25):
        super(SimplePusher, self).__init__()
        self.wrappedEnv = simple_push_v2.parallel_env(continuous_actions=False, max_cycles=maxStep)
        self.agent_count = 2
        self.observation_space = [Box(-inf, inf, (8,)), Box(-inf, inf, (19,))]
        self.action_space = [Discrete(5),Discrete(5)]

class SimpleSpread(GymEnvWrapper):
    def __init__(self,maxStep=25, discrete=True):
        super(SimpleSpread, self).__init__()
        self.wrappedEnv = simple_spread_v2.parallel_env(continuous_actions=~discrete, max_cycles=maxStep)
        self.agent_count = 3
        self.observation_space = [Box(-inf, inf, (18,)), Box(-inf, inf, (18,)), Box(-inf, inf, (18,))]
        self.action_space = [Discrete(5),Discrete(5),Discrete(5)] if discrete else \
                            [Box(0.0, 1.0, (5,)), Box(0.0, 1.0, (5,)), Box(0.0, 1.0, (5,))]
        self.discrete = discrete

from multiagent_particle_envs.make_env import make_env

class SimpleSpread_v3(GymEnvWrapper):
    def __init__(self):
        super(SimpleSpread_v3, self).__init__()
        self.wrappedEnv = make_env("simple_spread", benchmark=False)
        self.agent_count = 3
        self.observation_space = [Box(-inf, inf, (18,)), Box(-inf, inf, (18,)), Box(-inf, inf, (18,))]
        self.action_space = [Box(0.0, 1.0, (2,)), Box(0.0, 1.0, (2,)), Box(0.0, 1.0, (2,))]
        self.discrete = False
        self.owner = True

class SimpleWorldComm(GymEnvWrapper):
    def __init__(self,maxStep=25, discrete=True):
        super(SimpleWorldComm, self).__init__()
        self.wrappedEnv = simple_world_comm_v2.parallel_env(continuous_actions=~discrete, max_cycles=maxStep)
        self.agent_count = 6
        self.observation_space = [Box(-inf, inf, (34,)), Box(-inf, inf, (34,)), Box(-inf, inf, (34,)), Box(-inf, inf, (34,)),
                                  Box(-inf, inf, (28,)), Box(-inf, inf, (28,))]
        self.action_space = [Discrete(20),Discrete(5),Discrete(5),Discrete(5),Discrete(5),Discrete(5)] if discrete else \
                            [Box(0.0, 1.0, (9)), Box(0.0, 1.0, (5)), Box(0.0, 1.0, (5)), Box(0.0, 1.0, (5)), Box(0.0, 1.0, (5)), Box(0.0, 1.0, (5))]
        self.discrete = discrete

class SimpleTag(GymEnvWrapper):
    def __init__(self,maxStep=25, discrete=True):
        super(SimpleTag, self).__init__()
        self.wrappedEnv = simple_tag_v2.parallel_env(continuous_actions=~discrete, max_cycles=maxStep)
        self.agent_count = 4
        self.observation_space = [Box(-inf, inf, (16,)), Box(-inf, inf, (16,)), Box(-inf, inf, (16,)), Box(-inf, inf, (14,))]
        self.action_space = [Discrete(5),Discrete(5),Discrete(5),Discrete(5)] if discrete else \
            [Box(0.0, 1.0, (5,)), Box(0.0, 1.0, (5,)), Box(0.0, 1.0, (5,)), Box(0.0, 1.0, (5,)), Box(0.0, 1.0, (5,))]
        self.discrete = discrete

class SimpleCrypto(GymEnvWrapper):
    def __init__(self,maxStep=25):
        super(SimpleCrypto, self).__init__()
        self.wrappedEnv = simple_crypto_v2.parallel_env(continuous_actions=False, max_cycles=maxStep)
        self.agent_count = 3
        self.observation_space = [Box(-inf, inf, (4,)), Box(-inf, inf, (8,)), Box(-inf, inf, (8,))]
        self.action_space = [Discrete(4),Discrete(4),Discrete(4)]

class SimpleReference(GymEnvWrapper):
    def __init__(self,maxStep=25):
        super(SimpleReference, self).__init__()
        self.wrappedEnv = simple_reference_v2.parallel_env(continuous_actions=False, max_cycles=maxStep)
        self.agent_count = 2
        self.observation_space = [Box(-inf, inf, (21,)), Box(-inf, inf, (21,))]
        self.action_space = [Discrete(50),Discrete(50)]


if __name__ == '__main__':
    # import torch
    # from rl.utils.functions import onehot_from_logits
    # ped_env = SimpleAdversary()
    # obs = ped_env.reset()
    # print(ped_env.wrappedEnv)
    # is_done = [False]
    # while not is_done[0]:
    #     ped_env.render()
    #     next_obs, reward, is_done, info = ped_env.step(onehot_from_logits(torch.randn((3,5))))

    env = simple_reference_v2.parallel_env()
    print(env.observation_spaces)
    print(env.action_spaces)
