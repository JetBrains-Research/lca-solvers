from pipeline.data.composers.chain import Chunk, ComposerBlock
from pipeline.data.datapoint import Datapoint

from abc import ABC
from typing import Sequence, Type


class ChunkSorter(ComposerBlock, ABC):
    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from pipeline.data.composers.blocks.chunk_harvesting import ChunkHarvester
        return ChunkSorter, ChunkHarvester


class LexicographicSorter(ChunkSorter):
    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        return sorted(chunks, key=lambda c: c.rank)
