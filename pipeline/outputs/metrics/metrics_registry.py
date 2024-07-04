from typing import Callable

import torch

MetricName = str
MetricValue = int | float
MetricDefinition = Callable[
    [dict[str, torch.Tensor]],  # e.g. LMBatch
    MetricValue | torch.Tensor,  # tensor with metric values
]

METRICS_REGISTRY: dict[MetricName, MetricDefinition] = {
}
