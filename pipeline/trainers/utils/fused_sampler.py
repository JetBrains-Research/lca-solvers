import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class FusedSampler(Sampler[int]):
    def __init__(self,
                 start_sample_idx: int,
                 end_sample_idx: int,
                 dataset_length: int,
                 generator: torch.Generator | None = None,
                 ) -> None:
        super().__init__()

        self.start_sample_idx = start_sample_idx
        self.end_sample_idx = end_sample_idx
        self.dataset_length = dataset_length
        self.max_epochs = math.ceil(end_sample_idx / dataset_length)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        fused_weights = torch.rand(self.max_epochs, self.dataset_length, generator=self.generator)
        fused_indices = torch.argsort(fused_weights, dim=-1).flatten().tolist()
        yield from fused_indices[self.start_sample_idx:self.end_sample_idx]

    def __len__(self) -> int:
        return self.end_sample_idx - self.start_sample_idx
