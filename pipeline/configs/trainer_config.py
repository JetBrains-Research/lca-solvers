from pipeline.configs.config_base import ConfigBase
from pipeline.outputs.metrics.metrics_registry import MetricName

from dataclasses import dataclass


@dataclass
class FullFineTuningTrainerConfig(ConfigBase):
    # Iteration parameters
    max_iters: int
    valid_freq: int | None  # None means no validation at all
    gradient_accumulation_steps: int
    micro_batch_size: int

    # AdamW optimizer
    learning_rate: float
    beta_1: float
    beta_2: float
    weight_decay: float
    max_grad_norm: float

    # Cosine lr scheduler with warmup
    warmup_iters: int | None
    lr_decay_iters: int | None
    min_lr: float | None

    # Metrics (see METRICS_REGISTRY in pipeline/outputs/metrics/metrics_registry.py)
    train_metrics: list[MetricName]
    valid_metrics: list[MetricName]  # empty list means no validation at all

    # DataLoader
    shuffle: bool
    drop_last: bool
    num_workers: int
    random_seed: int | None
