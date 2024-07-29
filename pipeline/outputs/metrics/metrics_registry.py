from pipeline.outputs.metrics.cross_entropy import CrossEntropy, DetachedCrossEntropy
from pipeline.outputs.metrics.statistics import ema_factory, lazy_statistic_factory

METRICS_REGISTRY = {
    'cross_entropy': CrossEntropy,
    'detached_cross_entropy': DetachedCrossEntropy,
    'learning_rate': lazy_statistic_factory('learning_rate'),
}
METRICS_REGISTRY.update({
    f'ema_{name}': ema_factory(cls) for name, cls in METRICS_REGISTRY.items()
})
