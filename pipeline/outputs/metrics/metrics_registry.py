from pipeline.outputs.metrics.cross_entropy import CrossEntropy, DetachedCrossEntropy
from pipeline.outputs.metrics.metric_base import ema_factory

METRICS_REGISTRY = {
    'cross_entropy': CrossEntropy,
    'detached_cross_entropy': DetachedCrossEntropy,
}
METRICS_REGISTRY.update({
    f'ema_{name}': ema_factory(cls) for name, cls in METRICS_REGISTRY.items()
})
