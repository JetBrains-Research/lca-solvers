from pipeline.data.datapoint import Datapoint

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Sequence


@dataclass
class Chunk:
    content: str
    metadata: DefaultDict[str, Any] = field(default_factory=lambda: defaultdict(str))


class ChunkerMixin:
    @staticmethod
    def chunk(_datapoint: Datapoint) -> Sequence[Chunk]:
        return Chunk(content=''),


class FileGrainedChunker(ChunkerMixin):
    @staticmethod
    def chunk(datapoint: Datapoint) -> Sequence[Chunk]:
        return [
            Chunk(content=cnt, metadata=defaultdict(str, filename=fn))
            for fn, cnt in zip(*datapoint.repo_snapshot.values())
            # TODO: remove temporary hardcoded solution for data leakage
            if fn != 'tinygrad/llops/ops_llvm.py'
        ]


class LineGrainedChunker(ChunkerMixin):
    @staticmethod
    def chunk(datapoint: Datapoint) -> Sequence[Chunk]:
        return [
            Chunk(content=line, metadata=defaultdict(str, filename=fn))
            for fn, cnt in zip(*datapoint.repo_snapshot.values())
            # TODO: remove temporary hardcoded solution for data leakage
            if fn != 'tinygrad/llops/ops_llvm.py'
            for line in cnt.splitlines()
        ]
