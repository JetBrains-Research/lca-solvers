# TODO: grad norm, number of (masked) tokens, speed [tok/sec]?
import re
from abc import ABC, abstractmethod

import torch

StatisticName = str
StatisticValue = int | float


class StatisticBase(ABC):
    requires_tokenizer: bool = False

    @property
    def name(self) -> StatisticName:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', type(self).__name__).lower()

    def __repr__(self) -> str:
        return self.name

    def load_state(self, prev_value: StatisticValue | None) -> None:  # TODO: use
        pass  # default behavior

    @abstractmethod
    @torch.inference_mode
    def micro_batch_update(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def batch_commit(self, **kwargs) -> StatisticValue:
        raise NotImplementedError


class LazyStatistic(StatisticBase):
    def __init__(self, statistic_name: StatisticName) -> None:
        self.statistic_name = statistic_name
        self.value = None

    @property
    def name(self) -> StatisticName:
        return self.statistic_name

    def micro_batch_update(self, **kwargs) -> None:
        self.value = kwargs.get(self.statistic_name)

    def batch_commit(self, **_kwargs) -> StatisticValue:
        batch_statistic = self.value
        self.value = None
        return batch_statistic
