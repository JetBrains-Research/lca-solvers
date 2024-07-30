from pipeline.outputs.metrics.cross_entropy import CrossEntropy, DetachedCrossEntropy
from pipeline.outputs.metrics.metric_base import categorized_metric_factory
from pipeline.outputs.metrics.statistic_base import ema_factory, lazy_statistic_factory
from pipeline.outputs.metrics.counters import EpochCounter

METRICS_REGISTRY = {
    'cross_entropy': CrossEntropy,

    'detached_cross_entropy': DetachedCrossEntropy,
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
}
METRICS_REGISTRY.update({  # useless due to W&B native support :(
    f'ema_{name}': ema_factory(cls) for name, cls in METRICS_REGISTRY.items()
})
