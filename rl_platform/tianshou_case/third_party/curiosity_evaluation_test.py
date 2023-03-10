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

"""Simple test of the curiosity evaluation."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import torch

from rl_platform.tianshou_case.third_party import curiosity_evaluation
from rl_platform.tianshou_case.third_party import fake_gym_env
import numpy as np


def random_policy(unused_observation):
    action = np.random.randint(low=0, high=fake_gym_env.FakeGymEnv.NUM_ACTIONS)
    return action


class CuriosityEvaluationTest(unittest.TestCase):

    def EvalPolicy(self, policy):
        env = fake_gym_env.FakeGymEnv()

        # Distance between 2 consecutive curiosity rewards.
        reward_grid_size = 10.0

        # Times of evaluation.
        eval_time_steps = [100, 500, 3000]

        rewards = curiosity_evaluation.policy_state_coverage(
            env, policy, reward_grid_size, eval_time_steps)

        # The exploration reward is at most the number of steps.
        # It is equal to the number of steps when the policy explores a new state
        # at every time step.
        print('Curiosity reward: {}'.format(rewards))
        for k, r in rewards.items():
            self.assertGreaterEqual(k, r)

    def testRandomPolicy(self):
        self.EvalPolicy(random_policy)

    def testNNPolicy(self):
        batch_size = 1
        x = torch.randn(size=(batch_size,) + fake_gym_env.FakeGymEnv.OBSERVATION_SHAPE, dtype=torch.float32)
        x = torch.div(x, 255.0)

        # This is just to make the test run fast enough.
        # x_downscaled = torch.nn.functional.interpolate(x, [8, 8])
        # x_downscaled = torch.reshape(x_downscaled, [batch_size, -1])
        x_downscaled = torch.reshape(x, [batch_size, -1])

        # Logits to select the action.
        num_actions = 7
        net = torch.nn.Sequential(
            torch.nn.Linear(x_downscaled.shape[1], 32, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_actions, dtype=torch.float32)
        )

        y_logits = net(x_downscaled)
        temperature = 100.0
        y_logits /= temperature

        # Draw the action according to the distribution inferred by the logits.
        r = torch.randn(y_logits.shape)

        y_logits -= torch.log(-torch.log(r))
        y = torch.argmax(y_logits, dim=-1)

        input_state = torch.zeros(size=[37], dtype=torch.float32)
        output_state = input_state

        self.EvalPolicy(net)
