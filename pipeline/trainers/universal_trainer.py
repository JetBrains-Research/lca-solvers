from pipeline.model.adapters.adapter_base import AdapterBase
from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.checkpointers.data_structures import Checkpoint
from pipeline.outputs.loggers.logger_base import Log, LoggerBase
from pipeline.outputs.metrics.statistic_base import StatisticName, StatisticValue, StatisticBase
from pipeline.trainers.trainer_base import TrainerBase
from pipeline.trainers.utils.fused_sampler import FusedSampler
from pipeline.trainers.utils.schedulers import get_lr_from_cosine_scheduler_with_linear_warmup

import warnings
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
from transformers import PreTrainedTokenizerBase


class UniversalTrainer(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 train_ds: Dataset,
                 valid_ds: Dataset | None,
                 add_valid_ds: Dataset | None,
                 # auxiliary objects
                 adapter: AdapterBase,
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
                 train_metrics: list[StatisticBase],
                 valid_metrics: list[StatisticBase],
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
        self.adapter = adapter
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
        self.batch_size = gradient_accumulation_steps * micro_batch_size

        # environment
        self.is_on_cuda = (model.device.type == 'cuda')
        if random_seed is not None:
            torch.manual_seed(random_seed)
        torch.set_float32_matmul_precision(fp32_matmul_precision)
        logger.message(f"Set the FP32 matrix multiplication precision to '{fp32_matmul_precision}'.")

        # validation
        if valid_ds is None and add_valid_ds is not None:
            raise ValueError('Do not use an additional validation slot unless you have filled the first one.')
        if valid_ds is None and valid_freq is None and not valid_metrics:
            self.valid_freq = float('inf')
            self.valid_dl = None
            self.logger.message('Validation is disabled.')
        elif valid_ds is not None and valid_freq is not None and valid_metrics:
            self.valid_freq = valid_freq

            ds2dl = lambda x: DataLoader(
                dataset=x,
                batch_size=micro_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self.is_on_cuda,
                drop_last=False,
                prefetch_factor=prefetch_factor,
                persistent_workers=(valid_freq > max_iters),
                pin_memory_device=str(model.device),
            )

            self.valid_dl = ds2dl(valid_ds)
            self.add_valid_dl = ds2dl(add_valid_ds) if add_valid_ds is not None else None
        else:
            raise ValueError('The valid_ds, valid_freq and valid_metrics arguments do not match each other.')

        # training dataset
        sampler = FusedSampler(
            start_sample_idx=(self.batch_size * self.start_iter),
            end_sample_idx=(self.batch_size * max_iters),
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
        self.optimizer = self.adapter.init_optimizer(
            self.model, lr=learning_rate,
            betas=(beta_1, beta_2),
            weight_decay=weight_decay,
            fused=self.is_on_cuda)
        self.checkpointer.load_optimizer_state(self.optimizer)

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

    @torch.inference_mode
    def validate(self, valid_dl: DataLoader | None, verbose: bool = True) -> dict[StatisticName, StatisticValue]:
        if valid_dl is None:
            return {}

        is_additional = valid_dl is self.add_valid_dl
        training = self.model.training
        self.model.eval()

        valid_iter = tqdm(
            iterable=valid_dl,
            desc='Additional validation steps' if is_additional else 'Validation steps',
            position=1,
            leave=None,
            disable=not verbose,
        )

        for micro_batch in valid_iter:
            inputs = (
                input_ids, target_ids,
                loss_mask, completion_mask, category_ids,
                input_attn_mask, target_attn_mask,
                metadata,
            ) = tuple(t.to(self.model.device) for t in micro_batch.values())
            args, kwargs = self.adapter.get_args_kwargs(*inputs)

            model_output = self.model(*args, **kwargs)
            loss_per_token = F.cross_entropy(
                input=model_output.logits.flatten(0, 1),
                target=target_ids.flatten(0, 1),
                reduction='none',
            ).view_as(target_ids)

            locals_copy = locals().copy()
            locals_copy['trainer'] = locals_copy.pop('self')
            [metric.micro_batch_update(**locals_copy) for metric in self.valid_metrics]
            del locals_copy

        locals_copy = locals().copy()
        locals_copy['trainer'] = locals_copy.pop('self')
        valid_log = {
            f'{"additional_" if is_additional else ""}{metric.name}': metric.batch_commit(**locals_copy)
            for metric in self.valid_metrics
        }

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

        if self.start_iter == 0 and self.valid_dl is not None:
            valid_log = self.validate(self.valid_dl, verbose)
            valid_log |= self.validate(self.add_valid_dl, verbose)
            log = Log(iteration_number=0, valid_metrics=valid_log)
            self.logger.log(log)

        for iter_num in pbar_iter:
            pbar_accumulation.reset()

            learning_rate = self.get_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

            for _ in range(self.gradient_accumulation_steps):
                inputs = (
                    input_ids, target_ids,
                    loss_mask, completion_mask, category_ids,
                    input_attn_mask, target_attn_mask,
                    metadata,
                ) = tuple(t.to(self.model.device) for t in next(train_iter).values())
                args, kwargs = self.adapter.get_args_kwargs(*inputs)

                model_output = self.model(*args, **kwargs)
                loss_per_token = F.cross_entropy(
                    input=model_output.logits.flatten(0, 1),
                    target=target_ids.flatten(0, 1),
                    reduction='none',
                ).view_as(target_ids)
                # not accurate if drop_last=False and micro_batch_size != 1
                # see also PreprocessorBase.get_loss_mask comment in pipeline/data/preprocessors/preprocessor_base.py
                loss = loss_per_token[loss_mask].mean() / self.gradient_accumulation_steps

                self.grad_scaler.scale(loss).backward()

                locals_copy = locals().copy()
                locals_copy['trainer'] = locals_copy.pop('self')
                [metric.micro_batch_update(**locals_copy) for metric in self.train_metrics]
                del locals_copy

                pbar_accumulation.update()

            if self.max_grad_norm != 0:
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=self.adapter.get_trainable_parameters(self.model),
                    max_norm=self.max_grad_norm,
                )

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            locals_copy = locals().copy()
            locals_copy['trainer'] = locals_copy.pop('self')
            log = Log(iteration_number=iter_num + 1, train_metrics={
                metric.name: metric.batch_commit(**locals_copy) for metric in self.train_metrics
            })
            del locals_copy

            if (iter_num + 1) % self.valid_freq == 0:
                valid_log = self.validate(self.valid_dl, verbose)
                valid_log |= self.validate(self.add_valid_dl, verbose)
                log['valid_metrics'] = valid_log
            self.logger.log(log)

            if (iter_num + 1) % self.checkpointing_freq == 0:
                self.checkpointer.save_checkpoint(Checkpoint(
                    metrics=log,
                    model=self.model,
                    optimizer_state=self.optimizer.state_dict(),
                ))
