from pipeline.data.categories import CategoryType, CATEGORY2ID
from pipeline.outputs.metrics.statistic_base import StatisticValue, StatisticBase

from abc import ABC, abstractmethod
from enum import Enum
from typing import Type

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


def categorized_metric_factory(metric_cls: Type[MetricBase], category: CategoryType) -> Type[MetricBase]:
    class CategorizedMetric(metric_cls, ABC):
        def micro_batch_update(self, **kwargs) -> None:
            kwargs['loss_mask'] = (kwargs['category_ids'] == CATEGORY2ID[category])
            kwargs['loss'] = None
            return super().micro_batch_update(**kwargs)

    return CategorizedMetric
