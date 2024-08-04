from pipeline.data.composers.chain import File, Chunk, ComposerBlock
from pipeline.data.datapoint import Datapoint

from abc import ABC
from typing import Sequence, Type


class FileChunker(ComposerBlock, ABC):
    first_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from pipeline.data.composers.blocks.chunk_harvesting import ChunkHarvester
        from pipeline.data.composers.blocks.chunk_ranking import ChunkRanker
        return ChunkRanker, ChunkHarvester


class FileGrainedChunker(FileChunker):  # identity chunker
    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[Chunk]:
        return [Chunk(content=file.content, metadata=file.metadata, file_ref=file) for file in files]
