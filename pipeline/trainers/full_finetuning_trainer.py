from pipeline.outputs.checkpointing import CheckpointManager
from pipeline.outputs.loggers import LoggerBase
from pipeline.outputs.metrics.metrics_registry import MetricName
from pipeline.trainers.trainer_base import TrainerBase

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


class FullFineTuningTrainer(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 train_ds: Dataset,
                 valid_ds: Dataset | None,
                 # auxiliary objects
                 checkpointer: CheckpointManager,
                 logger: LoggerBase,
                 # iteration parameters
                 max_iters: int,
                 valid_freq: int,
                 gradient_accumulation_steps: int,
                 micro_batch_size: int,
                 # optimizer
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 weight_decay: float,
                 max_grad_norm: float,
                 # scheduler
                 decay_lr: bool,
                 warmup_iters: int | None,
                 lr_decay_iters: int | None,
                 min_lr: float | None,
                 # metrics
                 train_metrics: list[MetricName],
                 valid_metrics: list[MetricName],
                 # DataLoader
                 shuffle: bool,
                 drop_last: bool,
                 num_workers: int,
                 random_seed_dl: int | None,
                 # hardware
                 device: torch.device,
                 dtype: torch.dtype,
                 ) -> None:
        pass
