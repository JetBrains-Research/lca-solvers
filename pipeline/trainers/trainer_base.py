from pipeline.outputs.metrics.statistic_base import StatisticName, StatisticValue

from abc import ABC, abstractmethod

import torch


class TrainerBase(ABC):
    @abstractmethod
    @torch.inference_mode
    def validate(self, *args, **kwargs) -> dict[StatisticName, StatisticValue]:
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError
