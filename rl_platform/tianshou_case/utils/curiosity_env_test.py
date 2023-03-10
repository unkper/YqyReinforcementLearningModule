# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test of curiosity_env_wrapper.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from rl_platform.tianshou_case.utils import single_curiosity_env_wrapper as curiosity_env_wrapper
from rl_platform.tianshou_case.third_party import episodic_memory
import gym
import numpy as np


class DummyImageEnv(gym.Env):

    def __init__(self):
        self._num_actions = 4
        self._image_shape = (28, 28, 3)
        self._done_prob = 0.01

        self.action_space = gym.spaces.Discrete(self._num_actions)
        self.observation_space = gym.spaces.Box(
            0, 255, self._image_shape, dtype=np.uint8)

    def seed(self, seed=None):
        pass

    def step(self, action):
        observation = np.random.randint(0, 255, size=self._image_shape, dtype=np.uint8)
        reward = 0.0
        done = (np.random.rand() < self._done_prob)
        info = {}
        return observation, reward, done, info

    def reset(self, seed=None):
        self.seed(seed)
        return np.random.randint(0, 255, size=self._image_shape, dtype=np.uint8)

    def render(self, mode='human'):
        raise NotImplementedError('Rendering not implemented')



def embedding_similarity(x1, x2):
    assert x1.shape[0] == x2.shape[0]
    epsilon = 1e-6

    # Inner product between the embeddings in x1
    # and the embeddings in x2.
    s = np.sum(x1 * x2, axis=-1)

    s /= np.linalg.norm(x1, axis=-1) * np.linalg.norm(x2, axis=-1) + epsilon
    return 0.5 * (s + 1.0)


def linear_embedding(m, x):
    # Flatten all but the batch dimension if needed.
    if len(x.shape) > 2:
        x = np.reshape(x, [x.shape[0], -1])
    print(m.shape, "\n", x.shape)
    return np.matmul(x, m)


class EpisodicEnvWrapperTest(unittest.TestCase):

    def EnvFactory(self):
        return DummyImageEnv()

    def testResizeObservation(self):
        img_grayscale = np.random.randint(low=0, high=256, size=[64, 48, 1])
        img_grayscale = img_grayscale.astype(np.uint8)
        resized_img = curiosity_env_wrapper.resize_observation(img_grayscale,
                                                               [16, 12, 1])
        self.assertEqual((16, 12, 1), resized_img.shape)

        img_color = np.random.randint(low=0, high=256, size=[64, 48, 3])
        img_color = img_color.astype(np.uint8)
        resized_img = curiosity_env_wrapper.resize_observation(img_color,
                                                               [16, 12, 1])
        self.assertEqual((16, 12, 1), resized_img.shape)
        resized_img = curiosity_env_wrapper.resize_observation(img_color,
                                                               [16, 12, 3])
        self.assertEqual((16, 12, 3), resized_img.shape)

    def testEpisodicEnvWrapperSimple(self):
        vec_env = self.EnvFactory()

        embedding_size = 16
        vec_episodic_memory = episodic_memory.EpisodicMemory(
                            capacity=1000,
                            observation_shape=vec_env.observation_space.shape,
                            observation_compare_fn=embedding_similarity)


        mat = np.random.normal(size=[28 * 28 * 3, embedding_size])
        observation_embedding = lambda x, m=mat: linear_embedding(m, x)

        target_image_shape = (14, 14, 1)
        env_wrapper = curiosity_env_wrapper.CuriosityEnvWrapper(
            vec_env, vec_episodic_memory,
            observation_embedding,
            target_image_shape)

        observations = env_wrapper.reset()
        self.assertEqual(target_image_shape, observations.shape)

        dummy_actions = [1]
        for _ in range(10000):
            previous_mem_length = len(vec_episodic_memory)
            observation, unused_reward, done, unused_info = (
                env_wrapper.step(dummy_actions))
            current_mem_length = len(vec_episodic_memory)

            self.assertEqual(target_image_shape, observations.shape)
            if done:
                self.assertEqual(1, current_mem_length)
            else:
                self.assertGreaterEqual(current_mem_length,
                                        previous_mem_length)


