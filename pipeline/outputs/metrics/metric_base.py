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


def loss_based_metric_factory(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
    class LossBasedMetric(metric_cls, ABC):
        def micro_batch_update(self, **kwargs) -> None:
            kwargs['mask'] = kwargs['loss_mask']
            return super().micro_batch_update(**kwargs)

    return LossBasedMetric


def detached_metric_factory(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
    class DetachedMetric(metric_cls, ABC):
        def micro_batch_update(self, **kwargs) -> None:
            kwargs['mask'] = ~kwargs['loss_mask'] & kwargs['target_attn_mask']
            return super().micro_batch_update(**kwargs)

    return DetachedMetric


def completion_metric_factory(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
    class CompletionMetric(metric_cls, ABC):
        def micro_batch_update(self, **kwargs) -> None:
            kwargs['mask'] = kwargs['completion_mask']
            return super().micro_batch_update(**kwargs)

    return CompletionMetric


def context_metric_factory(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
    class ContextMetric(metric_cls, ABC):
        def micro_batch_update(self, **kwargs) -> None:
            kwargs['mask'] = ~kwargs['completion_mask'] & kwargs['target_attn_mask']
            return super().micro_batch_update(**kwargs)

    return ContextMetric


def full_metric_factory(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
    class FullMetric(metric_cls, ABC):
        def micro_batch_update(self, **kwargs) -> None:
            kwargs['mask'] = kwargs['target_attn_mask']
            return super().micro_batch_update(**kwargs)

    return FullMetric


def categorized_metric_factory(metric_cls: Type[MetricBase], category: CategoryType) -> Type[MetricBase]:
    class CategorizedMetric(metric_cls, ABC):
        def micro_batch_update(self, **kwargs) -> None:
            kwargs['mask'] = (kwargs['category_ids'] == CATEGORY2ID[category])
            return super().micro_batch_update(**kwargs)

    return CategorizedMetric
