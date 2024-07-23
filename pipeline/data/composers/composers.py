from pipeline.data.composers.base_composers import GrainedComposer
from pipeline.data.composers.chunking_mixins import FileGrainedChunker
from pipeline.data.composers.filtering_mixins import (
    InclusiveFileExtensionFilter,
    ExclusiveFileExtensionFilter,
)
from pipeline.data.composers.ranking_mixins import NegativePathDistance


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
