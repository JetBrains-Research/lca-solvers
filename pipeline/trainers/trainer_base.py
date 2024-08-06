from pipeline.outputs.metrics.metric_base import MetricName, MetricValue

from abc import ABC, abstractmethod

import torch


class TrainerBase(ABC):
    @abstractmethod
    @torch.inference_mode
    def validate(self, *args, **kwargs) -> dict[MetricName, MetricValue]:
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError