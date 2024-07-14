from abc import ABC, abstractmethod
from enum import Enum
from typing import Type

import torch

MetricName = str
MetricValue = int | float


class OptimizationMode(str, Enum):
    MIN = 'minimization'
    MAX = 'maximization'


class MetricBase(ABC):
    @property
    @abstractmethod
    def mode(self) -> OptimizationMode:
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode
    def micro_batch_update(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def batch_commit(self) -> MetricValue:
        raise NotImplementedError


def ema_factory(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
    class EMAMetric(metric_cls, ABC):
        def __init__(self, ema_alpha: float) -> None:
            super().__init__()
            self.ema_alpha = ema_alpha
            self.ema_state = None

        def batch_commit(self) -> MetricValue:
            batch_metric = super().batch_commit()
            if self.ema_state is None:
                self.ema_state = batch_metric
            else:
                self.ema_state += self.ema_alpha * (batch_metric - self.ema_state)
            return self.ema_state

    return EMAMetric
