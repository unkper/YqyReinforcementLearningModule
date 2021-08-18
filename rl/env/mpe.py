import gym
import torch

from gym.spaces import Box, Discrete
from numpy import inf

from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_speaker_listener_v3
from pettingzoo.mpe import simple_push_v2
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.utils import random_demo

class SimpleAdversary(gym.Env):
    def __init__(self, maxStep = 25):
        super(SimpleAdversary, self).__init__()
        self.wrappedEnv = simple_adversary_v2.parallel_env(N=2, max_cycles = maxStep, continuous_actions=False)
        self.agent_count = 3
        self.observation_space = [Box(-inf, inf, (8,)), Box(-inf, inf, (10,)), Box(-inf, inf, (10,))]
        self.action_space = [Discrete(5),Discrete(5),Discrete(5)]

    def step(self, action):
        import copy
        _action = copy.copy(action)
        _action = torch.argmax(_action,dim=1).detach().cpu().numpy()
        actions = {agent: _action[idx] for idx,agent in enumerate(self.wrappedEnv.agents)}
        obs, reward, is_done, info = self.wrappedEnv.step(actions)
        obs = list(obs.values())
        reward = list(reward.values())
        is_done = list(is_done.values())
        return obs, reward, is_done, "SimpleAdversary"

    def reset(self):
        obs = self.wrappedEnv.reset()
        obs = list(obs.values())
        return obs

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

class SimpleSpeakerListener(gym.Env):
    def __init__(self):
        super(SimpleSpeakerListener, self).__init__()
        self.wrappedEnv = simple_speaker_listener_v3.env(max_cycles=150, continuous_actions=False)
        self.agent_count = 2
        self.observation_space = [Box(-inf, inf, (3,)), Box(-inf, inf, (11,))]
        self.action_space = [Discrete(3),Discrete(5)]

    def step(self, action):
        import copy
        _action = copy.copy(action)
        _action = torch.argmax(_action,dim=1).detach().cpu().numpy()
        obs, reward, is_done = [], [], []
        for i in range(self.agent_count):
            self.wrappedEnv.step(_action[i])
            o, r, _isDone, _ = self.wrappedEnv.last()
            obs.append(o)
            reward.append(r)
            is_done.append(_isDone)
        if sum(is_done) > 0:
            is_done = [True for _ in range(self.agent_count)]
        return obs, reward, is_done, "SimpleSpeakerListener"

    def reset(self):
        self.wrappedEnv.reset()
        obs = []
        for agent in self.wrappedEnv.agents:
            obs.append(self.wrappedEnv.observe(agent))
        return obs

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

class SimplePusher(gym.Env):
    def __init__(self,maxStep=25):
        super(SimplePusher, self).__init__()
        self.wrappedEnv = simple_push_v2.parallel_env(continuous_actions=False, max_cycles=maxStep)
        self.agent_count = 2
        self.observation_space = [Box(-inf, inf, (8,)), Box(-inf, inf, (19,))]
        self.action_space = [Discrete(5),Discrete(5)]

    def step(self, action):
        import copy
        _action = copy.copy(action)
        _action = torch.argmax(_action,dim=1).detach().cpu().numpy()
        actions = {agent: _action[idx] for idx,agent in enumerate(self.wrappedEnv.agents)}
        obs, reward, is_done, info = self.wrappedEnv.step(actions)
        obs = list(obs.values())
        reward = list(reward.values())
        is_done = list(is_done.values())
        return obs, reward, is_done, "SimpleAdversary"

    def reset(self):
        obs = self.wrappedEnv.reset()
        obs = list(obs.values())
        return obs

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

class SimpleSpread(gym.Env):
    def __init__(self,maxStep=25):
        super(SimpleSpread, self).__init__()
        self.wrappedEnv = simple_spread_v2.parallel_env(continuous_actions=False, max_cycles=maxStep)
        self.agent_count = 3
        self.observation_space = [Box(-inf, inf, (18,)), Box(-inf, inf, (18,)), Box(-inf, inf, (18,))]
        self.action_space = [Discrete(5),Discrete(5),Discrete(5)]

    def step(self, action):
        import copy
        _action = copy.copy(action)
        _action = torch.argmax(_action,dim=1).detach().cpu().numpy()
        actions = {agent: _action[idx] for idx,agent in enumerate(self.wrappedEnv.agents)}
        obs, reward, is_done, info = self.wrappedEnv.step(actions)
        obs = list(obs.values())
        reward = list(reward.values())
        is_done = list(is_done.values())
        return obs, reward, is_done, "SimpleAdversary"

    def reset(self):
        obs = self.wrappedEnv.reset()
        obs = list(obs.values())
        return obs

    def render(self, mode='human'):
        self.wrappedEnv.render(mode)

    def seed(self, seed=None):
        self.wrappedEnv.seed(seed)

if __name__ == '__main__':
    import torch
    from rl.utils.functions import onehot_from_logits
    env = SimpleAdversary()
    obs = env.reset()
    print(env.wrappedEnv)
    is_done = [False]
    while not is_done[0]:
        next_obs, reward, is_done, info = env.step(onehot_from_logits(torch.randn((3,5))))
