import numpy as np
from tianshou.data import Batch


class DummyPolicy:
    def __init__(self, action_shape):
        self.action_shape = action_shape

    def __call__(self, batch: Batch, *args, **kwargs):
        batch_size = batch.obs.shape[0]
        batch.act = np.array([self.action_shape.sample() for _ in range(batch_size)])
        return batch

    def map_action_inverse(self, act):
        return act

    def exploration_noise(self, act, data):
        return act

    def map_action(self, act):
        return act
