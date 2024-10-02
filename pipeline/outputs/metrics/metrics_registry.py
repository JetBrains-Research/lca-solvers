from pipeline.outputs.metrics.counters import EpochCounter
from pipeline.outputs.metrics.cross_entropy import CrossEntropy
from pipeline.outputs.metrics.metric_base import (
    loss_based_metric_factory,
    detached_metric_factory,
    completion_metric_factory,
    context_metric_factory,
    full_metric_factory,
    categorized_metric_factory,
)
from pipeline.outputs.metrics.model_statistics import PastWeights
from pipeline.outputs.metrics.statistic_base import StatisticBase, ema_factory, lazy_statistic_factory
from pipeline.outputs.metrics.top_k_accuracy import top_k_accuracy_factory

from typing import Type


class MetricsRegistry(dict):
    def __getitem__(self, key: str) -> Type[StatisticBase]:
        if 'cross_entropy' in key:
            metric_cls = CrossEntropy
        elif 'top_' in key and key.endswith('_accuracy'):
            k = int(key.split('_')[-2])
            metric_cls = top_k_accuracy_factory(k)
        else:
            return dict.__getitem__(self, key)

        if key.startswith('detached_'):
            return detached_metric_factory(metric_cls)

        if key.startswith('completion_'):
            return completion_metric_factory(metric_cls)

        if key.startswith('context_'):
            return context_metric_factory(metric_cls)

        if key.startswith('full_'):
            return full_metric_factory(metric_cls)

        categorized_prefixes = (
            'commited_', 'common_', 'infile_',
            'inproject_', 'non_informative_',
            'random_', 'other_',
        )
        if key.startswith(categorized_prefixes):
            category = key.split('_')[0] if not key.startswith('non_informative_') else 'non_informative'
            return categorized_metric_factory(metric_cls, category)

        return loss_based_metric_factory(metric_cls)


METRICS_REGISTRY = MetricsRegistry({
    # in training only
    'epoch': EpochCounter,
    'learning_rate': lazy_statistic_factory('learning_rate'),
    'past_weights': PastWeights,  # compatible with SmoothPrefixUnmaskAdapter only
})
METRICS_REGISTRY.update({  # useless due to W&B native support :(
    f'ema_{name}': ema_factory(cls) for name, cls in METRICS_REGISTRY.items()
})
