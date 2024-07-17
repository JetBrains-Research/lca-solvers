from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import TRAINER_YAML
from pipeline.outputs.metrics.metric_base import MetricName

from dataclasses import dataclass
from typing import Literal


@dataclass
class FullFineTuningTrainerConfig(ConfigBase):
    _default_path = TRAINER_YAML

    # Iteration parameters
    max_iters: int
    valid_freq: int | None  # None means no validation at all
    checkpointing_freq: int | None  # None means no checkpointing at all
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
    ema_alpha: float  # (see ema_factory in pipeline/outputs/metrics/metric_base.py)
    # TODO: different ema_alpha for validation (consider valid_freq?)

    # DataLoader
    shuffle: bool
    drop_last: bool
    num_workers: int
    prefetch_factor: int
    random_seed: int | None

    # Floating point  TODO: another config for all global settings?
    fp32_matmul_precision: Literal['highest', 'high', 'medium']
