from pipeline.configs.config_base import ConfigBase
from pipeline.data.composers.chain import ComposerBlock

from dataclasses import dataclass, field
from typing import Sequence

from transformers import PreTrainedTokenizerBase


@dataclass
class ChainedComposerConfig(ConfigBase):
    pre_context_prompt: str
    post_context_prompt: str
    path_comment_template: str
    recalculate_random_category: bool
    blocks: Sequence[ComposerBlock] = field(default_factory=list)


@dataclass
class SplitComposerConfig(ConfigBase):
    max_block_size: int
    max_num_blocks: int
    block_sep: str
    recalculate_random_category: bool
    tokenizer: PreTrainedTokenizerBase
    blocks: Sequence[ComposerBlock] = field(default_factory=list)
