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
from pipeline.outputs.metrics.statistic_base import ema_factory, lazy_statistic_factory

METRICS_REGISTRY = {
    'cross_entropy': loss_based_metric_factory(CrossEntropy),

    'detached_cross_entropy': detached_metric_factory(CrossEntropy),
    'completion_cross_entropy': completion_metric_factory(CrossEntropy),
    'context_cross_entropy': context_metric_factory(CrossEntropy),
    'full_cross_entropy': full_metric_factory(CrossEntropy),
    'commited_cross_entropy': categorized_metric_factory(CrossEntropy, 'commited'),
    'common_cross_entropy': categorized_metric_factory(CrossEntropy, 'common'),
    'infile_cross_entropy': categorized_metric_factory(CrossEntropy, 'infile'),
    'inproject_cross_entropy': categorized_metric_factory(CrossEntropy, 'inproject'),
    'non_informative_cross_entropy': categorized_metric_factory(CrossEntropy, 'non_informative'),
    'random_cross_entropy': categorized_metric_factory(CrossEntropy, 'random'),
    'other_cross_entropy': categorized_metric_factory(CrossEntropy, 'other'),

    # in training only
    'epoch': EpochCounter,
    'learning_rate': lazy_statistic_factory('learning_rate'),
    'past_weights': PastWeights,  # compatible with SmoothPrefixUnmaskAdapter only
}
METRICS_REGISTRY.update({  # useless due to W&B native support :(
    f'ema_{name}': ema_factory(cls) for name, cls in METRICS_REGISTRY.items()
})
