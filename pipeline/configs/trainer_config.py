from pipeline.configs.config_base import ConfigBase
from pipeline.model.adapters.adapter_base import AdapterBase
from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.loggers.logger_base import LoggerBase
from pipeline.outputs.metrics.metric_base import MetricName

from dataclasses import dataclass
from typing import Literal

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class FullFineTuningTrainerConfig(ConfigBase):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    train_ds: Dataset
    valid_ds: Dataset | None

    # Auxiliary objects
    adapter: AdapterBase
    checkpointer: CheckpointManager
    logger: LoggerBase

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
    train_ema_alpha: float  # (see ema_factory in pipeline/outputs/metrics/statistic_base.py)
    valid_metrics: list[MetricName]  # empty list means no validation at all
    valid_ema_alpha: float | None  # if None, will be calculated as 1 - (1 - train_ema_alpha) ** valid_freq

    # DataLoader
    shuffle: bool
    drop_last: bool
    num_workers: int
    prefetch_factor: int | None
    random_seed: int | None

    # Floating point
    fp32_matmul_precision: Literal['highest', 'high', 'medium']
