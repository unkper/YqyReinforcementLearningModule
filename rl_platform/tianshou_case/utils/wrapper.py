from collections import deque, defaultdict
from typing import Optional

import gym
import numpy as np
import tianshou


class DisableRewardWrapper(gym.RewardWrapper):
    """
    Overview:
        Disable Env reward to Zero.
    Interface:
        ``__init__``, ``reward``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.

    """

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)

    def reward(self, reward):
        """
        Overview:
            Disable Env reward to Zero.
        Arguments:
            - reward(:obj:`Float`): Raw Reward
        Returns:
            - reward(:obj:`Float`): Zero Reward
        """
        return 0


class MarioRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
        """
        super().__init__(env)
        self.env = env

    def reward(self, reward):
        """
        Overview:
            Disable Env reward to Zero.
        Arguments:
            - reward(:obj:`Float`): Raw Reward
        Returns:
            - reward(:obj:`Float`): Zero Reward
        """
        return self.env.death_penalty_api


class FrameStackWrapper():
    """
    Overview:
       Stack latest n frames(usually 4 in Atari) as one observation.
    Interface:
        ``__init__``, ``reset``, ``step``, ``_get_ob``, ``new_shape``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - n_frame (:obj:`int`): the number of frames to stack.
        - ``observation_space``, ``frames``
    """

    def __init__(self, env, n_frames=4):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - n_frame (:obj:`int`): the number of frames to stack.
        """
        self.wrapper_env = env
        self._metadata: Optional[dict] = None
        self.n_frames = n_frames
        self.frames = defaultdict(lambda: deque([], maxlen=n_frames))
        self.render_mode = "rgb_array"
        obs_space = list(env.observation_spaces.values())[0]

        shape = (n_frames,) + obs_space.shape
        self.observation_spaces = {
            agentid: gym.spaces.Box(
                low=np.min(obs_space.low), high=np.max(obs_space.high), shape=shape, dtype=obs_space.dtype
            ) for agentid in self.wrapper_env.possible_agents}
        for agentid in self.wrapper_env.possible_agents:
            self.frames[agentid] = deque([], maxlen=n_frames)

    def reset(self, seed=None, options=None):
        """
        Overview:
            Resets the state of the environment and append new observation to frames
        Returns:
            - ``self._get_ob()``: observation
        """
        obs = self.wrapper_env.reset(seed=seed)
        for _ in range(self.n_frames):
            for agentid in self.wrapper_env.possible_agents:
                self.frames[agentid].append(obs[agentid])
        return self._get_ob()

    def step(self, action):
        """
        Overview:
            Step the environment with the given action. Repeat action, sum reward,  \
                and max over last observations, and append new observation to frames
        Arguments:
            - action (:obj:`Any`): the given action to step with.
        Returns:
            - ``self._get_ob()`` : observation
            - reward (:obj:`Any`) : amount of reward returned after previous action
            - done (:obj:`Bool`) : whether the episode has ended, in which case further \
                 step() calls will return undefined results
            - info (:obj:`Dict`) : contains auxiliary diagnostic information (helpful  \
                for debugging, and sometimes learning)
        """

        obs, reward, done, truncated, info = self.wrapper_env.step(action)
        for agentid in self.wrapper_env.possible_agents:
            self.frames[agentid].append(obs[agentid])
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        """
        Overview:
            The original wrapper use `LazyFrames` but since we use np buffer, it has no effect
        """
        obs = {}
        for agentid in self.wrapper_env.possible_agents:
            obs[agentid] = np.stack(self.frames[agentid], axis=0)
        return obs

    @property
    def metadata(self) -> dict:
        """Returns the environment metadata."""
        if self._metadata is None:
            return self.wrapper_env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def __getattr__(self, name):
        if name == "observation_spaces":
            return self.observation_spaces
        # 如果name是Foo的属性或者方法，就返回它
        if hasattr(self.wrapper_env, name):
            return getattr(self.wrapper_env, name)
        # 否则抛出异常
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")
