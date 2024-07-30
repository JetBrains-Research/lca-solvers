from pipeline.outputs.metrics.statistic_base import StatisticValue, StatisticBase

from typing import TypeVar

import torch

# avoid cyclical imports
FullFineTuningTrainer = TypeVar('FullFineTuningTrainer')


class EpochCounter(StatisticBase):
    def __init__(self) -> None:
        self.samples = 0
        self.ds_length = None

    def micro_batch_update(self, input_ids: torch.Tensor, trainer: FullFineTuningTrainer, **_kwargs) -> None:
        self.samples += input_ids.shape[0]
        self.ds_length = len(trainer.train_dl.dataset)

    def batch_commit(self) -> StatisticValue:
        return self.samples / self.ds_length
