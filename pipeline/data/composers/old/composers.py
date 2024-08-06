from pipeline.data.composers.composer_base import GrainedComposer
from pipeline.data.composers.old.chunking_mixins import (
    FileGrainedChunker,
    LineGrainedChunker, )
from pipeline.data.composers.old.filtering_mixins import (
    InclusiveFileExtensionFilter,
    ExclusiveFileExtensionFilter,
    PartialMemoryFilter,
    ChunkLengthFilter,
)
from pipeline.data.composers.old.ranking_mixins import (
    NegativePathDistance,
    InverseGroupingPathDistance,
)


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


class StripPathDistanceComposer(
    LineGrainedChunker, ChunkLengthFilter, NegativePathDistance, GrainedComposer):
    def __init__(self, min_len: int, max_len: int, *args, **kwargs) -> None:
        ChunkLengthFilter.__init__(self, min_len, max_len)
        GrainedComposer.__init__(self, *args, **kwargs)
