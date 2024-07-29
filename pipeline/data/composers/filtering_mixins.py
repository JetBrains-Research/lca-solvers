from pipeline.data.composers.chunking_mixins import Chunk
from pipeline.data.datapoint import Datapoint

import random
from typing import Sequence


class FilterMixin:
    @staticmethod
    def filter(chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        return chunks


class InclusiveFileExtensionFilter(FilterMixin):
    def __init__(self, whitelist: list[str]) -> None:
        self.whitelist = tuple(whitelist)

    def filter(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        return [chunk for chunk in chunks if chunk.metadata['filename'].endswith(self.whitelist)]


class ExclusiveFileExtensionFilter(FilterMixin):
    def __init__(self, blacklist: list[str]) -> None:
        self.blacklist = tuple(blacklist)

    def filter(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        return [chunk for chunk in chunks if not chunk.metadata['filename'].endswith(self.blacklist)]


class PartialMemoryFilter(FilterMixin):
    def __init__(self, dropout: float, random_seed: int | None) -> None:
        if not 0 <= dropout <= 1:
            raise ValueError('dropout must be selected from the interval [0, 1]. '
                             f'Got {dropout} instead.')
        self.dropout = dropout
        self.generator = random.Random(random_seed)

    def filter(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        return [chunk for chunk in chunks if self.generator.random() >= self.dropout]
