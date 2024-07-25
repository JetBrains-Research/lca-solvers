from pipeline.data.composed_datapoint import BatchComposedDatapoint

from abc import ABC, abstractmethod
from typing import TypedDict

import torch


class PreprocessedBatch(TypedDict):
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor
    category_ids: torch.Tensor
    attention_mask: torch.Tensor


class PreprocessorBase(ABC):
    @abstractmethod
    def get_loss_mask(self, *args, **kwargs) -> torch.Tensor:
        """
        Important note: different number of masked tokens in different
        micro-batches will break gradient accumulation, in which case
        the training loop should include corresponding gradient scaling.
        *or we just don't care :)
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, batch: BatchComposedDatapoint) -> PreprocessedBatch:
        raise NotImplementedError
