from pipeline.data.composers.chain import File, ComposerBlock
from pipeline.data.datapoint import Datapoint

from abc import ABC
from typing import Sequence, Type


class FilePreprocessor(ComposerBlock, ABC):
    first_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from pipeline.data.composers.blocks.file_chunking import FileChunker
        from pipeline.data.composers.blocks.file_filtering import FileFilter
        return FileFilter, FilePreprocessor, FileChunker


class EmptyLinesRemovalPreprocessor(FilePreprocessor):
    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        for file in files:
            file.content = '\n'.join(line for line in file.content.split('\n') if line.strip())
        return files
