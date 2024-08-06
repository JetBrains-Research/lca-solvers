from pipeline.configs.config_base import ConfigBase
from pipeline.data.composers.chain import ComposerBlock

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ChainedComposerConfig(ConfigBase):
    pre_context_prompt: str
    post_context_prompt: str
    path_comment_template: str
    blocks: Sequence[ComposerBlock] = field(default_factory=list)
