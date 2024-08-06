from pipeline.data.composers.old.chunking_mixins import Chunk
from pipeline.data.datapoint import Datapoint

import random
from collections import defaultdict
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


# shortcut for significant speed improvement; alternative is LinesHarvester
# TODO: better Mixins' decomposition
def merge_files(func):
    def decorated(*args, **kwargs) -> Sequence[Chunk]:
        files = defaultdict(list)
        for line in func(*args, **kwargs):
            files[line.metadata['filename']].append(line.content)

        return [
            Chunk(content='\n'.join(cnt), metadata=defaultdict(str, filename=fn))
            for fn, cnt in files.items()
        ]

    return decorated


class PartialMemoryFilter(FilterMixin):
    def __init__(self, dropout: float, random_seed: int | None) -> None:
        if not 0 <= dropout <= 1:
            raise ValueError('dropout must be selected from the interval [0, 1]. '
                             f'Got {dropout} instead.')
        self.dropout = dropout
        self.generator = random.Random(random_seed)

    @merge_files
    def filter(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        return [chunk for chunk in chunks if self.generator.random() >= self.dropout]


class ChunkLengthFilter(FilterMixin):
    def __init__(self, min_len: int, max_len: int) -> None:
        self.min_len = min_len
        self.max_len = max_len

    @merge_files
    def filter(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        for chunk in chunks:
            chunk.content = chunk.content.strip()  # TODO: better Mixins' decomposition
        return [chunk for chunk in chunks if self.min_len <= len(chunk.content) <= self.max_len]
