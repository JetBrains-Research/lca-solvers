from pipeline.data.composers.base_composers import ComposerBase
from pipeline.data.preprocessing.preprocessor_base import PreprocessorBase
from pipeline.outputs.checkpointing import ModelCheckpointBase
from pipeline.outputs.loggers import LoggerBase
from pipeline.trainers.trainer_base import TrainerBase

import torch
from datasets import Dataset


class FullFineTuningTrainer(TrainerBase):
    def __init__(self,
                 dataset: Dataset,
                 # main objects
                 composer: ComposerBase,
                 preprocessor: PreprocessorBase,
                 checkpointer: ModelCheckpointBase,
                 logger: LoggerBase,
                 # training config
                 max_iters: int,
                 valid_freq: int,
                 gradient_accumulation_steps: int,
                 micro_batch_size: int,
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 weight_decay: float,
                 max_grad_norm: float,
                 decay_lr: bool,
                 warmup_iters: int,
                 lr_decay_iters: int,
                 min_lr: float,
                 valid_size: int,
                 upper_bound_per_repo: int,
                 random_seed_split: int | None,
                 shuffle: bool,
                 drop_last: bool,
                 num_workers: int,
                 random_seed_dl: int | None,
                 device: torch.device,
                 dtype: torch.dtype,
                 compile: bool,  # noqa: built-in function that won't be used
                 ) -> None:
        pass
