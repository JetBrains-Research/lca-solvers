from __future__ import annotations

from pipeline.outputs.metrics.metric_base import MetricName, MetricValue, OptimizationMode
from pipeline.outputs.metrics.metrics_registry import METRICS_REGISTRY

import json
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch
from transformers import PreTrainedModel


class LoadingMode(str, Enum):
    SCRATCH = 'scratch'
    RESUME = 'resume'
    BEST = 'best'


@dataclass
class Checkpoint:
    iteration_number: int
    model: PreTrainedModel
    optimizer_state: dict
    metrics: dict[MetricName, MetricValue]


class CheckpointManager:
    def __init__(self,
                 init_from: LoadingMode | str,
                 saving_freq: int,
                 main_metric: MetricName,
                 directory: str,
                 checkpoint_directory_template: str,
                 extract_iteration_number: Callable[[str], int],
                 model_subdirectory: str,
                 optim_state_filename: str,
                 metrics_filename: str,
                 ) -> None:
        if main_metric not in METRICS_REGISTRY:
            raise ValueError('The specified main_metric is not contained in the registry.')

        self.init_from = init_from
        self.saving_freq = saving_freq  # TODO: use
        self.main_metric_name = main_metric
        self.main_metric = METRICS_REGISTRY[main_metric]
        self.directory = directory

        self._checkpoint_directory_template = checkpoint_directory_template
        self._extract_iteration_number = extract_iteration_number
        self._model_subdirectory = model_subdirectory
        self._optim_state_filename = optim_state_filename
        self._metrics_filename = metrics_filename

    def get_checkpoint_score(self, checkpoint_dir: str) -> MetricValue:
        metrics_file = os.path.join(checkpoint_dir, self._metrics_filename)
        with open(metrics_file) as stream:
            metrics = json.load(stream)

        metric_value = metrics.get(self.main_metric_name)

        if metric_value is None:
            raise RuntimeError(f'The {metrics_file} does not contain information '
                               'about the specified main_metric.')
        elif self.main_metric.mode == OptimizationMode.MIN:
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
                    key=self._extract_iteration_number,
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

    def get_iteration_number(self) -> int:
        checkpoint_dir = self.get_checkpoint_directory()
        if checkpoint_dir is not None:
            return self._extract_iteration_number(checkpoint_dir)
        else:
            return 0

    def get_model_subdirectory(self) -> str | None:
        checkpoint_dir = self.get_checkpoint_directory()
        if checkpoint_dir is not None:
            return os.path.join(checkpoint_dir, self._model_subdirectory)
        else:
            return None

    def init_optimizer(self, optimizer: torch.optim.AdamW) -> None:
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

        model_save_dir, optim_file, metrics_file = map(
            lambda x: os.path.join(checkpoint_dir, x),
            [self._model_subdirectory, self._optim_state_filename, self._metrics_filename],
        )

        checkpoint.model.save_pretrained(model_save_dir)
        torch.save(checkpoint.optimizer_state, optim_file)

        with open(metrics_file, 'w') as stream:
            json.dump(checkpoint.metrics, stream, indent=4)
