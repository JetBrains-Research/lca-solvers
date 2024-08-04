from pipeline.data.composers.chain import Chunk, ComposerBlock
from pipeline.data.datapoint import Datapoint

from abc import ABC
from typing import Sequence, Type


class ChunkHarvester(ComposerBlock, ABC):
    last_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from pipeline.data.composers.blocks.context_postprocessing import ContextPostprocessor
        return ContextPostprocessor,


class JoiningHarvester(ChunkHarvester):
    def __init__(self, chunks_sep: str) -> None:
        self.chunks_sep = chunks_sep

    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> str:
        return self.chunks_sep.join(chunk.content for chunk in chunks)


class PathCommentHarvester(JoiningHarvester):
    def __init__(self, chunks_sep: str, path_comment_template: str) -> None:
        super().__init__(chunks_sep)
        self.path_comment_template = path_comment_template

    def __call__(self, chunks: Sequence[Chunk], datapoint: Datapoint) -> str:
        for chunk in chunks:
            chunk.content = self.path_comment_template.format(
                filename=chunk.file_ref.metadata['filename'], content=chunk.content)
        return super().__call__(chunks, datapoint)
