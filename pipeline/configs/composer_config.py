from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import COMPOSER_YAML

from dataclasses import dataclass


@dataclass
class ComposerConfig(ConfigBase):
    _default_path = COMPOSER_YAML

    pre_context_prompt: str
    chunks_sep: str
    post_context_prompt: str
    path_comment_template: str


@dataclass
class InclusiveFileComposerConfig(ComposerConfig):
    whitelist: list[str]


@dataclass
class ExclusiveFileComposerConfig(ComposerConfig):
    blacklist: list[str]


@dataclass
class GroupingPathDistanceComposerConfig(ComposerConfig):
    ordered_groups: list[list[str]]
