from typing import Tuple, Any

import gym
import numpy as np
from gym.core import ObsType, ActType
from gym.wrappers import TimeLimit, FrameStack

from rl_platform.tianshou_case.third_party.single_curiosity_env_wrapper import resize_observation

target_image_shape = [80, 120, 3]
CAR_IMAGE_STACK = 4
CAR_ACTION_REAPET = 8

class WalkerEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = target_image_shape

    def reset(self, **kwargs):
        obs, dict = self.env.reset(**kwargs)
        obs = self.env.render()
        obs = resize_observation(obs, target_image_shape)
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, truncated, info = self.env.step(action)
        obs = self.env.render()
        obs = resize_observation(obs, target_image_shape)
        done = done or truncated
        return obs, rew, done, info


class CarRacingWrapper(gym.Wrapper):

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb, _ = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * CAR_IMAGE_STACK  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(CAR_ACTION_REAPET):
            img_rgb, reward, die, truncated, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == CAR_IMAGE_STACK
        return np.array(self.stack), total_reward, done, None

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


def create_walker_env(hardcore=False, max_step=2000, num_stack=4):
    env = WalkerEnvWrapper(
        FrameStack(
            TimeLimit(gym.make('BipedalWalker-v3', render_mode='rgb_array', hardcore=hardcore),
                      max_episode_steps=max_step), num_stack=num_stack))
    return env


def create_car_racing_env(max_step=2000, discrete=True):
    env = CarRacingWrapper(
        TimeLimit(gym.make("CarRacing-v2", render_mode='rgb_array', continuous=not discrete),
                  max_episode_steps=max_step))
    return env
