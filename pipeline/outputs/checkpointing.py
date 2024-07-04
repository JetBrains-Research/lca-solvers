from __future__ import annotations

from pipeline.outputs.metrics.metrics_registry import MetricName, MetricValue

from dataclasses import dataclass
from enum import Enum

import torch.nn as nn


@dataclass
class Checkpoint:
    model: nn.Module
    # TODO: optimizer, scheduler
    metrics: dict[MetricName, MetricValue]


    @staticmethod
    def load(path: str) -> Checkpoint:
        pass


class LoadingMode(str, Enum):
    HUB_MODEL = 'hugging_face_hub'
    BEST_MODEL = 'best_model'
    LAST_MODEL = 'last_model'


class CheckpointManager:
    def __init__(self,
                 init_from: LoadingMode,
                 freq: int,
                 directory: str,
                 tokenizer_name: str,
                 model_name: str,
                 ) -> None:
        self.init_from = init_from
        self.freq = freq
        self.directory = directory
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

    def get_model(self, loading_mode: LoadingMode | None = None) -> nn.Module:

        if loading_mode is None:
            return
