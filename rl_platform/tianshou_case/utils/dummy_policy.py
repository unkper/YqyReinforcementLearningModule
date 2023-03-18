from typing import Union

import numpy as np
from tianshou.data import Batch


class DummyPolicy:
    def __init__(self, action_shape, eps=1.0):
        self.action_shape = action_shape
        self._max_action_num = self.action_shape.n
        self.eps = eps

    def __call__(self, batch: Batch, *args, **kwargs):
        batch_size = batch.obs.shape[0]
        batch.act = np.array([self.action_shape.sample() for _ in range(batch_size)])
        return batch

    def map_action(self, act):
        return act

    def map_action_inverse(self, act):
        return act

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self._max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act


