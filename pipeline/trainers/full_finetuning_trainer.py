from pipeline.outputs.checkpointing import Checkpoint, CheckpointManager
from pipeline.outputs.loggers.logger_base import Log, LoggerBase
from pipeline.outputs.metrics.metric_base import MetricName, MetricValue
from pipeline.trainers.utils.fused_sampler import FusedSampler
from pipeline.trainers.utils.schedulers import get_lr_from_cosine_scheduler_with_linear_warmup

import warnings
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class FullFineTuningTrainer:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 train_ds: Dataset,
                 valid_ds: Dataset | None,
                 # auxiliary objects
                 checkpointer: CheckpointManager,
                 logger: LoggerBase,
                 # iteration parameters
                 max_iters: int,
                 valid_freq: int | None,
                 checkpointing_freq: int | None,
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
                 train_ema_alpha: float,
                 valid_metrics: list[MetricName],
                 valid_ema_alpha: float | None,
                 # DataLoader
                 shuffle: bool,
                 drop_last: bool,
                 num_workers: int,
                 prefetch_factor: int,
                 random_seed: int | None,
                 # Floating point
                 fp32_matmul_precision: Literal['highest', 'high', 'medium'],
                 ) -> None:
        # main objects
        self.model = model
        self.tokenizer = tokenizer
        self.checkpointer = checkpointer
        self.logger = logger

        # iterations
        self.checkpointing_freq = checkpointing_freq

        if checkpointing_freq is None:
            self.checkpointing_freq = float('inf')
            self.logger.message('Checkpointing is disabled.')
        elif valid_freq is not None and valid_freq != checkpointing_freq:
            warnings.warn('Validation and checkpointing are not synchronized (valid_freq != checkpointing_freq). '
                          'Resulting checkpoints will not contain validation metrics.')

        self.start_iter = checkpointer.get_iteration_number()
        self.max_iters = max_iters
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # environment
        self.is_on_cuda = (model.device.type == 'cuda')
        if random_seed is not None:
            torch.manual_seed(random_seed)
        torch.set_float32_matmul_precision(fp32_matmul_precision)
        logger.message(f"Set the FP32 matrix multiplication precision to '{fp32_matmul_precision}'.")

        # validation
        if valid_ds is None and valid_freq is None and not valid_metrics:
            self.valid_freq = float('inf')
            self.valid_dl = None
            self.logger.message('Validation is disabled.')
        elif valid_ds is not None and valid_freq is not None and valid_metrics:
            self.valid_freq = valid_freq
            self.valid_dl = DataLoader(
                dataset=valid_ds,
                batch_size=micro_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self.is_on_cuda,
                drop_last=False,
                prefetch_factor=prefetch_factor,
                persistent_workers=(valid_freq > max_iters),
                pin_memory_device=str(model.device),
            )

            if valid_ema_alpha is None:
                valid_ema_alpha = 1 - (1 - train_ema_alpha) ** valid_freq
                self.logger.message(f'valid_ema_alpha automatically set to {valid_ema_alpha:.05f}.')

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
            prefetch_factor=prefetch_factor,
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
        self.train_metrics = self.checkpointer.init_metrics('train_metrics', train_metrics, train_ema_alpha)
        self.valid_metrics = self.checkpointer.init_metrics('valid_metrics', valid_metrics, valid_ema_alpha)

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
    def validate(self, verbose: bool = True) -> dict[MetricName, MetricValue]:  # TODO: zero step
        training = self.model.training
        self.model.eval()

        valid_iter = tqdm(
            iterable=self.valid_dl,
            desc='Validation steps',
            position=1,
            leave=None,
            disable=not verbose,
        )

        for micro_batch in valid_iter:
            micro_batch = [t.to(self.model.device) for t in micro_batch.values()]
            input_ids, target_ids, loss_mask, category_ids, attention_mask = micro_batch

            model_output = self.model(input_ids, attention_mask=attention_mask)

            locals_copy = locals().copy()
            locals_copy.pop('self')
            [metric.micro_batch_update(**locals_copy) for metric in self.valid_metrics.values()]
            del locals_copy

        valid_log = {name: metric.batch_commit() for name, metric in self.valid_metrics.items()}

        self.model.train(training)
        return valid_log

    def train(self, verbose: bool = True) -> None:
        self.model.train()

        train_iter = iter(self.train_dl)
        pbar_iter = trange(
            self.start_iter, self.max_iters,
            desc='Optimization steps',
            initial=self.start_iter,
            total=self.max_iters,
            position=0,
            disable=not verbose,
        )
        pbar_accumulation = trange(
            self.gradient_accumulation_steps,
            desc='Gradient accumulation steps',
            position=1,
            leave=None,
            disable=not verbose,
        )

        for iter_num in pbar_iter:
            pbar_accumulation.reset()

            lr = self.get_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            for _ in range(self.gradient_accumulation_steps):
                micro_batch = [t.to(self.model.device) for t in next(train_iter).values()]
                input_ids, target_ids, loss_mask, category_ids, attention_mask = micro_batch

                model_output = self.model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(model_output.logits[loss_mask], target_ids[loss_mask])
                loss = loss / self.gradient_accumulation_steps

                self.grad_scaler.scale(loss).backward()

                locals_copy = locals().copy()
                locals_copy.pop('self')
                [metric.micro_batch_update(**locals_copy) for metric in self.train_metrics.values()]
                del locals_copy

                pbar_accumulation.update()

            if self.max_grad_norm != 0:
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            log = Log(
                iteration_number=iter_num,
                train_metrics={name: metric.batch_commit() for name, metric in self.train_metrics.items()},
            )
            if (iter_num + 1) % self.valid_freq == 0:
                log['valid_metrics'] = self.validate(verbose)
            self.logger.log(log)

            if (iter_num + 1) % self.checkpointing_freq == 0:
                self.checkpointer.save_checkpoint(Checkpoint(
                    metrics=log,
                    model=self.model,
                    optimizer_state=self.optimizer.state_dict(),
                ))
