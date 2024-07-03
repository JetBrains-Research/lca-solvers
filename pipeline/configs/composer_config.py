from pipeline.configs.config_base import Config

from dataclasses import dataclass


@dataclass
class ComposerConfig(Config):
    pre_context_prompt: str
    chunks_sep: str
    post_context_prompt: str
    path_comment_template: str
