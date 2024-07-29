from pipeline.outputs.metrics.statistics import StatisticValue, StatisticBase

from abc import ABC, abstractmethod
from enum import Enum

MetricName = str
MetricValue = StatisticValue


class OptimizationMode(str, Enum):
    MIN = 'minimization'
    MAX = 'maximization'


class MetricBase(StatisticBase, ABC):
    @property
    @abstractmethod
    def mode(self) -> OptimizationMode:
        raise NotImplementedError
