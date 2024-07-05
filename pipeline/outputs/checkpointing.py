from __future__ import annotations

from pipeline.configs.checkpointing_config import CheckpointManagerConfig
from pipeline.outputs.metrics.metrics_registry import MetricName, MetricValue, METRICS_REGISTRY

import json
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from torch.optim import AdamW


class LoadingMode(str, Enum):
    SCRATCH = 'scratch'
    RESUME = 'resume'
    BEST = 'best'


@dataclass
class Checkpoint:
    iteration_number: int
    model_state: OrderedDict[str, torch.Tensor]
    optimizer_state: dict
    metrics: dict[MetricName, MetricValue]


class CheckpointManager:
    def __init__(self,
                 init_from: LoadingMode | str,
                 saving_freq: int,
                 main_metric: MetricName,
                 directory: str,
                 checkpoint_directory_template: str,
                 model_state_filename: str,
                 optim_state_filename: str,
                 metrics_filename: str,
                 ) -> None:
        if main_metric not in METRICS_REGISTRY:
            raise ValueError('The specified main_metric is not contained in the registry.')

        self.init_from = init_from
        self.saving_freq = saving_freq
        self.main_metric = METRICS_REGISTRY[main_metric]
        self.directory = directory

        self._checkpoint_directory_template = checkpoint_directory_template
        self._model_state_filename = model_state_filename
        self._optim_state_filename = optim_state_filename
        self._metrics_filename = metrics_filename

    def get_checkpoint_score(self, checkpoint_dir: str) -> MetricValue:
        metrics_file = os.path.join(checkpoint_dir, self._metrics_filename)
        with open(metrics_file) as stream:
            metrics = json.load(stream)

        metric_value = metrics.get(self.main_metric.name)

        if metric_value is None:
            raise RuntimeError(f'The {metrics_file} does not contain information '
                               'about the specified main_metric.')
        elif self.main_metric.mode == 'minimization':
            return metric_value
        else:
            return -metric_value

    def get_checkpoint_directory(self) -> str | None:
        match self.init_from:
            case LoadingMode.SCRATCH:
                return None
            case LoadingMode.RESUME:
                return max(
                    os.listdir(self.directory),
                    key=CheckpointManagerConfig.extract_iteration_number,
                    default=None,
                )
            case LoadingMode.BEST:
                return min(
                    os.listdir(self.directory),
                    key=self.get_checkpoint_score,
                    default=None,
                )
            case _:  # user-defined checkpoint directory
                return self.init_from

    def get_iteration_number(self) -> int | None:
        checkpoint_dir = self.get_checkpoint_directory()
        if checkpoint_dir is not None:
            return CheckpointManagerConfig.extract_iteration_number(checkpoint_dir)
        else:
            return None

    def init_model(self, model: nn.Module) -> None:
        checkpoint_dir = self.get_checkpoint_directory()
        if checkpoint_dir is not None:
            model_file = os.path.join(checkpoint_dir, self._model_state_filename)
            model.load_state_dict(torch.load(model_file))

    def init_optimizer(self, optimizer: AdamW) -> None:
        checkpoint_dir = self.get_checkpoint_directory()
        if checkpoint_dir is not None:
            optim_file = os.path.join(checkpoint_dir, self._optim_state_filename)
            optimizer.load_state_dict(torch.load(optim_file))

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        checkpoint_dir = os.path.join(
            self.directory,
            self._checkpoint_directory_template.format(
                iteration_number=checkpoint.iteration_number),
        )

        if os.path.exists(checkpoint_dir):
            warnings.warn(f'The contents of the checkpoint {checkpoint_dir} have been overwritten.')

        model_file, optim_file, metrics_file = map(
            lambda x: os.path.join(checkpoint_dir, x),
            [self._model_state_filename, self._optim_state_filename, self._metrics_filename],
        )

        torch.save(checkpoint.model_state, model_file)
        torch.save(checkpoint.optimizer_state, optim_file)

        with open(metrics_file, 'w') as stream:
            json.dump(checkpoint.metrics, stream, indent=4)
