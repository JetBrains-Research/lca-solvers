# TODO: grad norm, number of (masked) tokens, speed [tok/sec]?
from abc import ABC, abstractmethod
from typing import Type

import torch

StatisticValue = int | float


class StatisticBase(ABC):
    def reinit(self, prev_value: StatisticValue | None) -> None:
        pass  # default behavior

    @abstractmethod
    @torch.inference_mode
    def micro_batch_update(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def batch_commit(self, **kwargs) -> StatisticValue:
        raise NotImplementedError


def ema_factory(statistic_cls: Type[StatisticBase]) -> Type[StatisticBase]:
    class EMAStatistic(statistic_cls, ABC):
        def __init__(self, ema_alpha: float) -> None:
            super().__init__()
            self.ema_alpha = ema_alpha
            self.ema_state = None

        def reinit(self, ema_state: float | None) -> None:
            self.ema_state = ema_state

        def batch_commit(self, **kwargs) -> StatisticValue:
            batch_metric = super().batch_commit(**kwargs)
            if self.ema_state is None:
                self.ema_state = batch_metric
            else:
                self.ema_state += self.ema_alpha * (batch_metric - self.ema_state)
            return self.ema_state

    return EMAStatistic


def lazy_statistic_factory(statistic_name: str) -> Type[StatisticBase]:
    class LazyStatistic(StatisticBase):
        def __init__(self) -> None:
            self.value = None

        def micro_batch_update(self, **kwargs) -> None:
            self.value = kwargs.get(statistic_name)

        def batch_commit(self, **_kwargs) -> StatisticValue:
            batch_statistic = self.value
            self.value = None
            return batch_statistic

    return LazyStatistic
