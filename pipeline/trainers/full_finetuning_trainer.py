from pipeline.outputs.checkpointing import CheckpointManager
from pipeline.outputs.loggers import LoggerBase
from pipeline.outputs.metrics.metrics_registry import MetricName, MetricValue
from pipeline.trainers.trainer_base import TrainerBase
from pipeline.trainers.utils.fused_sampler import FusedSampler
from pipeline.trainers.utils.schedulers import get_lr_from_cosine_scheduler_with_linear_warmup

from functools import partial

import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase


# TODO: refactor
# TODO: test determinism and everything else
# TODO: set_float32_matmul_precision
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
                 valid_freq: int | None,
                 gradient_accumulation_steps: int,
                 micro_batch_size: int,
                 # optimizer
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 weight_decay: float,
                 max_grad_norm: float,
                 # scheduler
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
                 random_seed: int | None,
                 ) -> None:
        # main objects
        self.model = model
        self.tokenizer = tokenizer
        self.checkpointer = checkpointer
        self.logger = logger  # TODO

        # iterations
        self.start_iter = checkpointer.get_iteration_number()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_batch_size = micro_batch_size

        # environment
        self.is_on_cuda = (model.device.type == 'cuda')
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # validation
        if valid_ds is None and valid_freq is None and not valid_metrics:
            self.valid_freq = float('inf')
            self.valid_dl = None
        elif valid_ds is not None and valid_freq is not None and valid_metrics:
            self.valid_freq = valid_freq
            self.valid_dl = DataLoader(
                dataset=valid_ds,
                batch_size=micro_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self.is_on_cuda,
                drop_last=False,
                persistent_workers=(valid_freq > max_iters),
                pin_memory_device=str(model.device),
            )
        else:
            raise ValueError('The valid_ds, valid_freq and valid_metrics arguments do not match each other.')

        # training dataset
        batch_size = gradient_accumulation_steps * micro_batch_size
        sampler = FusedSampler(
            start_sample_idx=(batch_size * self.start_iter),
            end_sample_idx=(batch_size * max_iters),
            dataset_length=len(train_ds),
        ) if shuffle else None

        self.train_dl = DataLoader(
            dataset=train_ds,
            batch_size=micro_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.is_on_cuda,
            drop_last=drop_last,
            pin_memory_device=str(model.device),
        )

        # optimizer initialization
        self.optimizer = self._init_adamw(learning_rate, beta_1, beta_2, weight_decay)

        # gradient utilities
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(model.dtype == torch.float16))
        self.max_grad_norm = max_grad_norm

        # scheduler initialization
        if warmup_iters is None and lr_decay_iters is None and min_lr is None:
            self.get_lr = lambda _: learning_rate
        elif warmup_iters is not None and lr_decay_iters is not None and min_lr is not None:
            self.get_lr = partial(
                get_lr_from_cosine_scheduler_with_linear_warmup,
                min_lr=min_lr,
                max_lr=learning_rate,
                warmup_iters=warmup_iters,
                lr_decay_iters=lr_decay_iters,
            )
        else:
            raise ValueError('The warmup_iters, lr_decay_iters and min_lr arguments do not match each other.')

        # metrics
        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics

    def _init_adamw(self,
                    learning_rate: float,
                    beta_1: float,
                    beta_2: float,
                    weight_decay: float,
                    ) -> torch.optim.AdamW:
        decay_params = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay_params = [p for p in self.model.parameters() if p.dim() < 2]
        params = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0},
        ]

        optimizer = torch.optim.AdamW(
            params=params,
            lr=learning_rate,
            betas=(beta_1, beta_2),
            fused=self.is_on_cuda)
        self.checkpointer.init_optimizer(optimizer)

        return optimizer

    @torch.inference_mode
    def validate(self) -> dict[MetricName, MetricValue]:
        assert self.valid_dl is not None  # TODO: remove
        training = self.model.training
        self.model.eval()

        pass  # TODO

        self.model.train(training)

        return ...  # TODO

    def train(self) -> None:
        pass  # TODO
