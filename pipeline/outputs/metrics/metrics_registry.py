from pipeline.outputs.metrics.counters import EpochCounter
from pipeline.outputs.metrics.cross_entropy import CrossEntropy
from pipeline.outputs.metrics.model_statistics import PastWeights
from pipeline.outputs.metrics.statistic_base import LazyStatistic
from pipeline.outputs.metrics.top_k_accuracy import TopKAccuracy

METRICS_REGISTRY = {
    # metrics
    'cross_entropy': CrossEntropy,
    'top_k_accuracy': TopKAccuracy,

    # statistics
    'epoch': EpochCounter,
    'lazy_statistic': LazyStatistic,
    'past_weights': PastWeights,  # compatible with SmoothPrefixUnmaskAdapter only
}
