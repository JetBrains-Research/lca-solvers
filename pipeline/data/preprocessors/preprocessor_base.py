from pipeline.data.composed_datapoint import BatchComposedDatapoint

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, TypedDict
from typing_extensions import Self

import torch
from torch.utils.data._utils.collate import default_collate_fn_map  # noqa: see below


class BatchMetadata(dict):
    def __getitem__(self, key: Any) ->  Any:
        if not isinstance(key, str):
            return self
        else:
            return super().__getitem__(key)

    def to(self, *args, **kwargs) -> Self:
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(*args, **kwargs)

        return self


class PreprocessedBatch(TypedDict):
    input_ids: torch.Tensor
    target_ids: torch.Tensor

    loss_mask: torch.Tensor
    completion_mask: torch.Tensor
    category_ids: torch.Tensor

    input_attn_mask: torch.Tensor
    target_attn_mask: torch.Tensor

    metadata: BatchMetadata[str, Any]


# although this is hidden from users, the PyTorch documentation includes such an example:
# https://pytorch.org/docs/stable/data.html#torch.utils.data._utils.collate.collate
default_collate_fn_map[BatchMetadata] = lambda x, *args, **kwargs: x[0]


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
    def get_completion_mask(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_category_ids(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, batch: BatchComposedDatapoint) -> PreprocessedBatch:
        raise NotImplementedError


class AmortizedPreprocessorBase(PreprocessorBase, ABC):
    def __init__(self, num_chars_per_token: int, verbose: bool) -> None:
        self.num_chars_per_token = num_chars_per_token
        self.verbose = verbose

    def _inc_num_chars_per_token(self) -> None:
        old_value = self.num_chars_per_token
        self.num_chars_per_token = math.ceil(1.5 * self.num_chars_per_token)

        if self.verbose:
            warnings.warn(
                f'num_chars_per_token has been increased from {old_value} to {self.num_chars_per_token} '
                'due to an underestimation of the length of the truncated character sequence.')
