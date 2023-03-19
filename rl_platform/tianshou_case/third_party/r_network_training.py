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

"""Set of functions used to train a R-network."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import logging
import os
import random

import torch
import torch as th
from tensorboardX import SummaryWriter
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.env import SubprocVectorEnv
from torch.optim import RAdam

from rl_platform.tianshou_case.net.r_network import RNetwork
from rl_platform.tianshou_case.third_party.constants import Const
import numpy as np

from rl_platform.tianshou_case.utils.dummy_policy import DummyPolicy


def generate_positive_example(buffer_position,
                              next_buffer_position):
    """Generates a close enough pair of states."""
    first = buffer_position
    second = next_buffer_position

    # Make R-network symmetric.
    # Works for DMLab (navigation task), the symmetry assumption might not be
    # valid for all the environments.
    if random.random() < 0.5:
        first, second = second, first
    return first, second


def generate_negative_example(buffer_position,
                              len_episode_buffer,
                              max_action_distance):
    """Generates a far enough pair of states."""
    assert buffer_position < len_episode_buffer
    # Defines the interval that must be excluded from the sampling.
    time_interval = (Const.NEGATIVE_SAMPLE_MULTIPLIER * max_action_distance)
    min_index = max(buffer_position - time_interval, 0)
    max_index = min(buffer_position + time_interval + 1, len_episode_buffer)

    # Randomly select an index outside the interval.
    effective_length = len_episode_buffer - (max_index - min_index)
    range_max = effective_length - 1
    if range_max <= 0:
        return buffer_position, None
    index = random.randint(0, range_max)
    if index >= min_index:
        index = max_index + (index - min_index)
    return buffer_position, index


def compute_next_buffer_position(buffer_position,
                                 positive_example_candidate,
                                 max_action_distance,
                                 mode):
    """Computes the buffer position for the next training example."""
    if mode == 'v3_affect_num_training_examples_overlap':
        # This version was initially not intended (changing max_action_distance
        # affects the number of training examples, and we can also get overlap
        # across generated examples), but we have it because it produces good
        # results (reward at ~40 according to raveman@ on 2018-10-03).
        # R-nets /cns/vz-d/home/dune/episodic_curiosity/raphaelm_train_r_mad2_4 were
        # generated with this version (the flag was set
        # v1_affect_num_training_examples, but it referred to a "buggy" version of
        # v1 that is reproduced here with that v3).
        return buffer_position + random.randint(1, max_action_distance) + 1
    if mode == 'v1_affect_num_training_examples':
        return positive_example_candidate + 1
    if mode == 'v2_fixed_num_training_examples':
        # Produces the ablation study in the paper submitted to ICLR'19
        # (https://openreview.net/forum?id=SkeK3s0qKQ), section S4.1.
        return buffer_position + random.randint(1, 5) + 1


def create_training_data_from_episode_buffer_v4(episode_buffer,
                                                max_action_distance,
                                                avg_num_examples_per_env_step):
    """Sampling of positive/negative examples without using stride logic."""
    num_examples = int(avg_num_examples_per_env_step * len(episode_buffer))
    num_examples_per_class = num_examples // 2
    # We first generate positive pairs, and then sample from them (ensuring that
    # we don't select twice exactly the same pair (i,i+j)).
    positive_pair_candidates = []
    for first in range(len(episode_buffer)):
        for j in range(1, max_action_distance + 1):
            second = first + j
            if second >= len(episode_buffer):
                continue
            positive_pair_candidates.append(
                (first, second) if random.random() > 0.5 else (second, first))
    assert len(positive_pair_candidates) >= num_examples_per_class
    positive_pairs = random.sample(positive_pair_candidates,
                                   num_examples_per_class)

    # Generate negative pairs.
    num_negative_candidates = len(episode_buffer) * (
            len(episode_buffer) -
            2 * Const.NEGATIVE_SAMPLE_MULTIPLIER * max_action_distance) / 2
    # Make sure we have enough negative examples to sample from (with some
    # headroom). If that does not happen (meaning very short episode buffer given
    # current values of negative_sample_multiplier, max_action_distance), don't
    # generate any training example.
    if num_negative_candidates < 2 * num_examples_per_class:
        return [], [], []
    negative_pairs = set()
    while len(negative_pairs) < num_examples_per_class:
        i = random.randint(0, len(episode_buffer) - 1)
        j = generate_negative_example(
            i, len(episode_buffer), max_action_distance)[1]
        # Checking this is not strictly required, because it should happen
        # infrequently with current parameter values.
        # We still check for it for the symmetry with the positive example case.
        if (i, j) not in negative_pairs and (j, i) not in negative_pairs:
            negative_pairs.add((i, j))

    x1 = []
    x2 = []
    labels = []
    for i, j in positive_pairs:
        x1.append(episode_buffer[i])
        x2.append(episode_buffer[j])
        labels.append(1)
    for i, j in negative_pairs:
        x1.append(episode_buffer[i])
        x2.append(episode_buffer[j])
        labels.append(0)
    return x1, x2, labels


def create_training_data_from_episode_buffer_v123(episode_buffer,
                                                  max_action_distance,
                                                  mode):
    """Samples intervals and forms pairs."""
    first_second_label = []
    buffer_position = 0
    while True:
        positive_example_candidate = (
                buffer_position + random.randint(1, max_action_distance))
        next_buffer_position = compute_next_buffer_position(
            buffer_position, positive_example_candidate,
            max_action_distance, mode)

        if (next_buffer_position >= len(episode_buffer) or
                positive_example_candidate >= len(episode_buffer)):
            break
        label = random.randint(0, 1)
        if label:
            first, second = generate_positive_example(buffer_position,
                                                      positive_example_candidate)
        else:
            first, second = generate_negative_example(buffer_position,
                                                      len(episode_buffer),
                                                      max_action_distance)
        if first is None or second is None:
            break
        first_second_label.append((first, second, label))
        buffer_position = next_buffer_position
    x1 = []
    x2 = []
    labels = []
    for first, second, label in first_second_label:
        x1.append(episode_buffer[first])
        x2.append(episode_buffer[second])
        labels.append(label)
    return x1, x2, labels


class RNetworkTrainer(object):
    """Train R network in an online way."""

    def __init__(self,
                 r_model: RNetwork,
                 observation_history_size=20000,
                 training_interval=20000,
                 batch_size=32,
                 num_train_epochs=6,
                 checkpoint_dir=None,
                 lr = 1e-3,
                 writer: SummaryWriter = None,
                 device='cuda' if th.cuda.is_available() else 'cpu'):
        # The training interval is assumed to be the same as the history size
        # for invalid negative values.
        if training_interval < 0:
            training_interval = observation_history_size

        self._r_model: RNetwork = r_model
        if device == 'cuda':
            self._r_model.cuda()
        self._training_interval = training_interval
        self._batch_size = batch_size
        self._num_train_epochs = num_train_epochs

        # Keeps track of the last N observations.
        # Those are used to train the R network in an online way.
        self._fifo_observations = [None] * observation_history_size
        self._fifo_dones = [None] * observation_history_size
        self._fifo_index = 0
        self._fifo_count = 0
        self.device = device
        self._writer = writer
        self.optimizer = RAdam(self._r_model.parameters(), lr=lr)

        # Used to save checkpoints.
        self._current_epoch = 0
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            checkpoint_period_in_epochs = self._num_train_epochs
            self._save_dir = os.path.join(checkpoint_dir)

    def on_new_observation(self, observation, unused_reward, done, info):
        """Event triggered when the environments generate a new observation."""
        if len(observation.shape) >= 3 or info is None or 'frame' not in info:
            self._fifo_observations[self._fifo_index] = observation
        else:
            # Specific to Parkour (stores velocity, joints as the primary
            # observation).
            self._fifo_observations[self._fifo_index] = info['frame']
        self._fifo_dones[self._fifo_index] = done
        self._fifo_index = (
                (self._fifo_index + 1) % len(self._fifo_observations))
        self._fifo_count += 1

        if (self._fifo_count > 0 and
                self._fifo_count % self._training_interval == 0):
            logging.info('Training the R-network after: {}'.format(
                self._fifo_count))
            history_observations, history_dones = self._get_flatten_history()
            if len(history_observations) > 1:
                self.train(history_observations, history_dones)

    def _get_flatten_history(self):
        """Convert the history given as a circular fifo to a linear array."""
        if self._fifo_count < len(self._fifo_observations):
            return (self._fifo_observations[:self._fifo_count],
                    self._fifo_dones[:self._fifo_count])

        # Reorder the indices.
        history_observations = self._fifo_observations[self._fifo_index:]
        history_observations.extend(self._fifo_observations[:self._fifo_index])
        history_dones = self._fifo_dones[self._fifo_index:]
        history_dones.extend(self._fifo_dones[:self._fifo_index])
        return history_observations, history_dones

    def _split_history(self, observations, dones):
        """Returns some individual trajectories."""
        if len(observations) == 0:  # pylint: disable=g-explicit-length-test
            return []

        # Number of environments that generated "observations",
        # and total number of steps.
        nsteps = len(dones)

        # Starting index of the current trajectory.
        start_index = 0

        trajectories = []
        for k in range(nsteps):
            if dones[k] or k == nsteps - 1:
                next_start_index = k + 1
                time_slice = observations[start_index:next_start_index]
                trajectories.append([obs for obs in time_slice])
                start_index = next_start_index

        return trajectories

    def _prepare_data(self, observations, dones):
        """Generate the positive and negative pairs used to train the R network."""
        max_action_distance = 5
        mode = 'v2_fixed_num_training_examples'

        all_x1 = []
        all_x2 = []
        all_labels = []
        trajectories = self._split_history(observations, dones)
        for trajectory in trajectories:
            x1, x2, labels = create_training_data_from_episode_buffer_v123(
                trajectory, max_action_distance, mode)
            all_x1.extend(x1)
            all_x2.extend(x2)
            all_labels.extend(labels)

        return all_x1, all_x2, all_labels

    def _shuffle(self, x1, x2, labels):
        sample_count = len(x1)
        assert len(x2) == sample_count
        assert len(labels) == sample_count
        permutation = np.random.permutation(sample_count)
        x1 = [x1[p] for p in permutation]
        x2 = [x2[p] for p in permutation]
        labels = [labels[p] for p in permutation]
        return x1, x2, labels

    def train(self, history_observations, history_dones):
        """Do one pass of training of the R-network."""
        x1, x2, labels = self._prepare_data(history_observations, history_dones)
        x1, x2, labels = self._shuffle(x1, x2, labels)

        # Split between train and validation data.
        n = len(x1)
        train_count = n
        x1_train, x2_train, labels_train = (
            th.as_tensor(x1[:train_count], device=self.device, dtype=th.float32),
            th.as_tensor(x2[:train_count], device=self.device, dtype=th.float32),
            th.as_tensor(labels[:train_count], device=self.device, dtype=th.int64))
        x1_valid, x2_valid, labels_valid = (
            th.as_tensor(x1[train_count:], device=self.device, dtype=th.float32),
            th.as_tensor(x2[train_count:], device=self.device, dtype=th.float32),
            th.as_tensor(labels[train_count:], device=self.device, dtype=th.int64))

        validation_data = ([th.as_tensor(x1_valid, device=self.device, dtype=th.float32),
                            th.as_tensor(x2_valid, device=self.device, dtype=th.float32)],
                           th.nn.functional.one_hot(
                               th.as_tensor(labels_valid, device=self.device, dtype=th.int64),
                               num_classes=2))

        # TODO(damienv): model checkpointing of R-network should be done
        # in ppo2.py, check whether it includes the R-network part.
        # Otherwise, we would train but never saves the model.
        logs, loss = self._r_model.fit([x1_train, x2_train], labels_train,
                                       batch_size=self._batch_size,
                                       epochs=self._num_train_epochs,
                                       validation_data=validation_data,
                                       optimizer=self.optimizer)
        self._writer.add_scalar("r_training/loss", loss, self._current_epoch)
        # Note: the same could possibly be achieved using parameters "callback",
        # "initial_epoch", "epochs" in fit_generator. However, this is not really
        # clear how this initial epoch is supposed to work.
        # TODO(damienv): change it to use callbacks of fit_generator.
        # 保存目前的训练结果
        for _ in range(self._num_train_epochs):
            self._current_epoch += 1
            # if self._checkpointer is not None:
            #     self._checkpointer.on_epoch_end(self._current_epoch)
        if self._save_dir is not None:
            th.save(self._r_model, os.path.join(self._save_dir, "r_network_weight_{}.pt".format(self._current_epoch)))


from easydict import EasyDict


def train_r_network_with_collector(env_func, file_name, params: EasyDict, load_file=None):
    writer = SummaryWriter(file_name)

    policy = DummyPolicy(env_func().action_space)
    train_envs = SubprocVectorEnv([env_func for _ in range(params.train_env_num)])
    buffer = VectorReplayBuffer(1000, len(train_envs))
    collector = Collector(policy, train_envs, buffer=buffer)

    net = RNetwork(params.target_image_shape, device=params.set_device)
    if load_file is not None:
        net = torch.load(load_file, map_location="cuda")
    if params.set_device == 'cuda':
        net = net.cuda()
    r_trainer = RNetworkTrainer(
        net,
        observation_history_size=params.observation_history_size,
        training_interval=params.training_interval,
        num_train_epochs=params.num_train_epochs,
        checkpoint_dir=file_name,
        device=params.set_device,
        batch_size=params.batch_size,
        writer=writer)
    from tqdm import tqdm

    pbar = tqdm(total=params.total_feed_step, desc="r-network training:")
    i = 0

    while i < params.total_feed_step:
        collector.collect(n_step=params.step_interval)
        batch, _ = buffer.sample(params.step_interval)
        for j in range(params.step_interval):
            r_trainer.on_new_observation(batch.obs[j], batch.rew[j], batch.done[j], batch.info[j])
        pbar.update(params.step_interval)
        i += params.step_interval
