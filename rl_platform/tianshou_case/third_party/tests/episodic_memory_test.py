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

"""Test of episodic_memory.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import unittest

import torch

from rl_platform.tianshou_case.third_party import episodic_memory
from rl_platform.tianshou_case.net.r_network import RNetwork
import numpy as np

from rl_platform.tianshou_case.third_party.episodic_memory import similarity_to_memory

net = torch.load(
    r"D:\projects\python\PedestrainSimulationModule\rl_platform\tianshou_case\mario\r_network\Mario_v3_PPO_2023_03_18_16_59_14\r_network_weight_1000.pt")
net = net.cuda()
net = net.eval()


def embedding_similarity(x1, x2):
    assert x1.shape[0] == x2.shape[0]
    epsilon = 1e-6

    # Inner product between the embeddings in x1
    # and the embeddings in x2.
    s = np.sum(x1 * x2, axis=-1)

    s /= np.linalg.norm(x1, axis=-1) * np.linalg.norm(x2, axis=-1) + epsilon
    return 0.5 * (s + 1.0)


class EpisodicMemoryTest(unittest.TestCase):

    def RunTest(self, memory, observation_shape, add_count):
        expected_size = min(add_count, memory.capacity)

        for _ in range(add_count):
            observation = net.embed_observation(np.random.normal(size=[1, ] + observation_shape))
            memory.add(observation, dict())
        self.assertEqual(expected_size, len(memory))

        current_observation = net.embed_observation(np.random.normal(size=[1, ] + observation_shape))
        similarities = memory.similarity(current_observation)
        self.assertEqual(expected_size, len(similarities))
        self.assertLessEqual(similarities.all(), 1.0)
        self.assertGreaterEqual(similarities.all(), 0.0)
        similar = similarity_to_memory(current_observation,
                                       memory)
        pprint.pprint(similar)

    def testEpisodicMemory(self):
        observation_shape = [4, 42, 42]
        embed_observation_shape = [512]
        memory = episodic_memory.EpisodicMemory(
            observation_shape=embed_observation_shape,
            observation_compare_fn=net.embedding_similarity,
            capacity=150)

        self.RunTest(memory,
                     observation_shape,
                     add_count=100)
        memory.reset()

        self.RunTest(memory,
                     observation_shape,
                     add_count=200)
        memory.reset()
