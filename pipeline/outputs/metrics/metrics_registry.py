from dataclasses import dataclass
from typing import Callable, Literal

import torch

MetricName = str
MetricValue = int | float


@dataclass
class MetricMetainfo:
    name: MetricName
    definition: Callable[
        [dict[str, torch.Tensor]],  # e.g. LMBatch
        MetricValue | torch.Tensor,  # tensor with metric values
    ]
    mode: Literal['minimization', 'maximization']


METRICS_REGISTRY: dict[MetricName, MetricMetainfo] = {
    'cross_entropy': ...,  # TODO
}
