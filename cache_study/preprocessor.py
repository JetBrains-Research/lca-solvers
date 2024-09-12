from enum import Enum


class BOSUsage(str, Enum):
    DISABLED = 'disabled'
    IN_EACH_BLOCK = 'in_each_block'
    HEAD_BLOCKS_ONLY = 'head_blocks_only'
    TAIL_SEAM = 'tail_seam'


class PreprocessorMixin:
    pass  # TODO
