from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class ComposerConfig(ConfigBase):
    pre_context_prompt: str
    chunks_sep: str
    post_context_prompt: str
    path_comment_template: str
