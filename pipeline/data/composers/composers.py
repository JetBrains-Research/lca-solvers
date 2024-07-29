from pipeline.data.composers.base_composers import GrainedComposer
from pipeline.data.composers.chunking_mixins import (
    FileGrainedChunker,
    LineGrainedChunker, Chunk,
)
from pipeline.data.composers.filtering_mixins import (
    InclusiveFileExtensionFilter,
    ExclusiveFileExtensionFilter,
    PartialMemoryFilter,
)
from pipeline.data.composers.ranking_mixins import (
    NegativePathDistance,
    InverseGroupingPathDistance,
)

from collections import defaultdict
from typing import Sequence


class PathDistanceComposer(FileGrainedChunker, NegativePathDistance, GrainedComposer):
    pass


class InclusiveFileExtensionPathDistanceComposer(
    FileGrainedChunker, InclusiveFileExtensionFilter, NegativePathDistance, GrainedComposer):
    def __init__(self, whitelist: list[str], *args, **kwargs) -> None:
        InclusiveFileExtensionFilter.__init__(self, whitelist)
        GrainedComposer.__init__(self, *args, **kwargs)


class ExclusiveFileExtensionPathDistanceComposer(
    FileGrainedChunker, ExclusiveFileExtensionFilter, NegativePathDistance, GrainedComposer):
    def __init__(self, blacklist: list[str], *args, **kwargs) -> None:
        ExclusiveFileExtensionFilter.__init__(self, blacklist)
        GrainedComposer.__init__(self, *args, **kwargs)


class GroupingPathDistanceComposer(
    FileGrainedChunker, InclusiveFileExtensionFilter, InverseGroupingPathDistance, GrainedComposer):
    def __init__(self, ordered_groups: list[list[str]], *args, **kwargs) -> None:
        whitelist = [extension for group in ordered_groups for extension in group]
        InclusiveFileExtensionFilter.__init__(self, whitelist)
        InverseGroupingPathDistance.__init__(self, ordered_groups)
        GrainedComposer.__init__(self, *args, **kwargs)


class PartialMemoryPathDistanceComposer(
    LineGrainedChunker, PartialMemoryFilter, NegativePathDistance, GrainedComposer):
    def __init__(self, dropout: float, random_seed: int | None, *args, **kwargs) -> None:
        PartialMemoryFilter.__init__(self, dropout, random_seed)
        GrainedComposer.__init__(self, *args, **kwargs)

    # shortcut for significant speed improvement; alternative is LinesHarvester
    def filter(self, *args, **kwargs) -> Sequence[Chunk]:
        files = defaultdict(list)
        for line in super().filter(*args, **kwargs):
            files[line.metadata['filename']].append(line.content)

        return [
            Chunk(content='\n'.join(cnt), metadata=defaultdict(str, filename=fn))
            for fn, cnt in files.items()
        ]
