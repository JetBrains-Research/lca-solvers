from pipeline.outputs.checkpointers.data_structures import LoadingMode, Checkpoint
from pipeline.outputs.loggers.logger_base import Log
from pipeline.outputs.metrics.metric_base import MetricName, MetricValue, OptimizationMode, MetricBase
from pipeline.outputs.metrics.metrics_registry import METRICS_REGISTRY

import json
import os
import warnings

from typing import Callable, Literal

import torch


class CheckpointManager:  # aka checkpointer
    def __init__(self,
                 init_from: LoadingMode | str,
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
        self.main_metric_name = main_metric
        self.main_metric = METRICS_REGISTRY[main_metric]
        self.directory = directory

        self._checkpoint_directory_template = checkpoint_directory_template
        self._extract_iteration_number = extract_iteration_number
        self._model_subdirectory = model_subdirectory
        self._optim_state_filename = optim_state_filename
        self._metrics_filename = metrics_filename

    def get_wandb_resume_mode(self) -> Literal['allow', 'never'] | None:
        match self.init_from:
            case LoadingMode.SCRATCH:
                return None
            case LoadingMode.RESUME:
                return 'allow'
            case _:
                return 'never'

    def load_metrics(self, checkpoint_dir: str) -> Log:
        metrics_file = os.path.join(checkpoint_dir, self._metrics_filename)
        with open(metrics_file) as stream:
            return Log(**json.load(stream))

    def get_checkpoint_score(self, checkpoint_dir: str) -> MetricValue:
        checkpoint_dir = os.path.join(self.directory, checkpoint_dir)
        metrics = self.load_metrics(checkpoint_dir)
        metric_value = metrics.get('valid_metrics', metrics['train_metrics']).get(self.main_metric_name)

        if metric_value is None:
            raise RuntimeError(f'The {checkpoint_dir} does not contain information '
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
                    next(os.walk(self.directory))[1],
                    key=self._extract_iteration_number,
                    default=None,
                )
            case LoadingMode.BEST:
                return min(
                    next(os.walk(self.directory))[1],
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
            return os.path.join(self.directory, checkpoint_dir, self._model_subdirectory)
        else:
            return None

    def load_optimizer_state(self, optimizer: torch.optim.AdamW) -> None:
        checkpoint_dir = self.get_checkpoint_directory()
        if checkpoint_dir is not None:
            optim_file = os.path.join(self.directory, checkpoint_dir, self._optim_state_filename)
            optimizer.load_state_dict(torch.load(optim_file))

    def init_metrics(self,
                     group: Literal['train_metrics', 'valid_metrics'],
                     metrics: list[MetricName],
                     ema_alpha: float | None = None,
                     ) -> dict[MetricName, MetricBase]:
        checkpoint_dir = self.get_checkpoint_directory()
        metrics_dict = dict()

        if checkpoint_dir is None:
            metrics_states = dict()
        else:
            checkpoint_dir = os.path.join(self.directory, checkpoint_dir)
            metrics_states = self.load_metrics(checkpoint_dir)[group]

        for name in metrics:
            init_args = [ema_alpha] if name.startswith('ema_') else list()
            metrics_dict[name] = METRICS_REGISTRY[name](*init_args)
            metrics_dict[name].reinit(metrics_states.get(name))

        return metrics_dict

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        checkpoint_dir = os.path.join(
            self.directory,
            self._checkpoint_directory_template.format(
                iteration_number=checkpoint.metrics['iteration_number']),
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
