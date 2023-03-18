import logging
import os

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace

from rl_platform.tianshou_case.standard_gym.wrapper import RewardType
from rl_platform.tianshou_case.utils.common import save_video, rgb2gray

target_image_shape = [4, 42, 42]
IMAGE_STACK = 4
ACTION_REPEAT = 4
action_type = [["right"], ["right", "A"]]


def resize_images(images):
    # 将输入图像从CHW形式转换为HWC形式
    input_img_hwc = np.transpose(images, (1, 2, 0))
    # 计算新的图像尺寸
    new_height, new_width = int(target_image_shape[1]), int(target_image_shape[2])
    # 缩放图像
    resized_img_hwc = cv2.resize(input_img_hwc, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_img_hwc


class MarioWrapper(gym.Wrapper):
    def __init__(self, env, discrete_reward: RewardType = RewardType.ZERO_REWARD):
        super().__init__(env)
        self._d_reward: RewardType = discrete_reward

    def reset(self):
        self.counter = 0

        self.die = False
        img_rgb = self.env.reset()
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * IMAGE_STACK  # four frames for decision
        return np.transpose(resize_images(np.array(self.stack)), (2, 0, 1))

    def task_reward(self, die, img_rgb, reward):
        return reward

    def step(self, action):
        done = False
        truncated = False
        total_reward = 0
        total_task_r = 0
        for i in range(ACTION_REPEAT):
            img_rgb, reward, done, truncated, _ = self.env.step(action)
            # reward model
            if self._d_reward == RewardType.ZERO_REWARD:
                # give zero reward for icm and ec
                reward = 0
                task_reward = self.task_reward(done, img_rgb, reward)
            elif self._d_reward == RewardType.ADV_REWARD:
                logging.error("暂不支持")
                reward = self.task_reward(done, img_rgb, reward)
                task_reward = reward
            else:
                task_reward = reward
            total_reward += reward
            total_task_r += task_reward
            if done:
                break
            img_gray = rgb2gray(img_rgb)
            self.stack.pop(0)
            self.stack.append(img_gray)
            assert len(self.stack) == IMAGE_STACK
        obs = np.transpose(resize_images(np.array(self.stack)), (2, 0, 1))
        return obs, total_reward, done, truncated, {"task_reward": total_task_r}


def create_mario_env(reward_type: RewardType = RewardType.RAW_REWARD, level="1-2"):
    level = "SuperMarioBros-{}-v3".format(level)
    return MarioWrapper(JoypadSpace(gym_super_mario_bros.make(level), action_type), discrete_reward=reward_type)


if __name__ == '__main__':
    env = create_mario_env()
    state = env.reset()

    done = False
    obs_arr = []
    for episode in range(2):
        while True:
            if done:
                state = env.reset()
                done = False
                break
            state, reward, done, truncated, info = env.step(env.action_space.sample())
            obs_arr.append(state)
        # pprint.pprint(state.shape)
        # env.render(mode="human")
    save_path = os.path.abspath("./animation.mp4")
    save_video(save_path, obs_arr)

    env.close()
