from pipeline.outputs.loggers.logger_base import Log

from dataclasses import dataclass
from enum import Enum

import torch.nn as nn


class LoadingMode(str, Enum):
    SCRATCH = 'scratch'
    RESUME = 'resume'
    BEST = 'best'


@dataclass
class Checkpoint:
    metrics: Log
    model: nn.Module
    optimizer_state: dict
