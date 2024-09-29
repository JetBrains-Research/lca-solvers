from pipeline.data.composers.blocks.chunk_harvesting import (
    JoiningHarvester,
    PathCommentHarvester,
)
from pipeline.data.composers.blocks.chunk_ranking import (
    NegativePathDistanceRanker,
    FunctionCallRanker,
    FileExtensionRanker,
    RandomRanker,
)
from pipeline.data.composers.blocks.chunk_sorting import (
    LexicographicSorter,
)
from pipeline.data.composers.blocks.context_postprocessing import (
    PartialMemoryPostprocessor,
    LineLengthPostprocessor,
    LineStripPostprocessor,
    InverseFrequencyMemoryPostprocessor,
)
from pipeline.data.composers.blocks.file_chunking import (
    FileGrainedChunker,
    CodeSegmentGrainedChunker,
)
from pipeline.data.composers.blocks.file_filtering import (
    NullFileFilter,
    InclusiveFileExtensionFilter,
    ExclusiveFileExtensionFilter,
    EmptyFileFilter,
    FileLengthFilter,
    TokenizedFileLengthFilter,
    CharTokenRatioFilter,
)
from pipeline.data.composers.blocks.file_preprocessing import (
    EmptyLinesRemovalPreprocessor,
    DeclarationOnlyPreprocessor,
)

BLOCKS_REGISTRY = {
    # file_filtering
    'null_file_filter': NullFileFilter,
    'inclusive_file_extension_filter': InclusiveFileExtensionFilter,
    'exclusive_file_extension_filter': ExclusiveFileExtensionFilter,
    'empty_file_filter': EmptyFileFilter,
    'file_length_filter': FileLengthFilter,
    'tokenized_file_length_filter': TokenizedFileLengthFilter,
    'char_token_ratio_filter': CharTokenRatioFilter,

    # file_preprocessing
    'empty_lines_removal_preprocessor': EmptyLinesRemovalPreprocessor,
    'declaration_only_preprocessor': DeclarationOnlyPreprocessor,

    # file_chunking
    'file_grained_chunker': FileGrainedChunker,
    'code_segment_grained_chunker': CodeSegmentGrainedChunker,

    # chunk_ranking
    'negative_path_distance_ranker': NegativePathDistanceRanker,
    'function_call_ranker': FunctionCallRanker,
    'file_extension_ranker': FileExtensionRanker,
    'random_ranker': RandomRanker,

    # chunk_sorting
    'lexicographic_sorter': LexicographicSorter,

    # chunk_harvesting
    'joining_harvester': JoiningHarvester,
    'path_comment_harvester': PathCommentHarvester,

    # context_postprocessing
    'partial_memory_postprocessor': PartialMemoryPostprocessor,
    'line_length_postprocessor': LineLengthPostprocessor,
    'line_strip_postprocessor': LineStripPostprocessor,
    'inverse_frequency_memory_postprocessor': InverseFrequencyMemoryPostprocessor,
}
