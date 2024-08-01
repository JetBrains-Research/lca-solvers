from pipeline.outputs.loggers.logger_base import Log

from dataclasses import dataclass
from enum import Enum

from transformers import PreTrainedModel


class LoadingMode(str, Enum):
    SCRATCH = 'scratch'
    RESUME = 'resume'
    BEST = 'best'


@dataclass
class Checkpoint:
    metrics: Log
    model: PreTrainedModel
    optimizer_state: dict
