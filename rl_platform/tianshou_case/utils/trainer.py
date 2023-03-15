from typing import Any, Callable, Dict, Optional, Union, Tuple

import numpy as np

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger, LazyLogger


class OnpolicyTrainer(BaseTrainer):

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Optional[Collector],
        max_epoch: int,
        step_per_epoch: int,
        repeat_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            learning_type="onpolicy",
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            step_per_collect=step_per_collect,
            episode_per_collect=episode_per_collect,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            test_in_train=test_in_train,
            **kwargs,
        )

    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one on-policy update."""
        assert self.train_collector is not None
        losses = self.policy.update(
            0,
            self.train_collector.buffer,
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )
        self.train_collector.reset_buffer(keep_statistics=True)
        step = max([1] + [len(v) for v in losses.values() if isinstance(v, list)])
        self.gradient_step += step
        self.log_update_data(data, losses)

    def train_step(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        data, result, stop_fn_flag = super().train_step()




def onpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OnpolicyTrainer run method.

    It is identical to ``OnpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OnpolicyTrainer(*args, **kwargs).run()


onpolicy_trainer_iter = OnpolicyTrainer
