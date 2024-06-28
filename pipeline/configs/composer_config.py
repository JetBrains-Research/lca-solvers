from __future__ import annotations
from dataclasses import dataclass, asdict

import yaml


@dataclass
class ComposerConfig:
    pre_context_prompt: str
    chunks_sep: str
    post_context_prompt: str
    path_comment_template: str

    dict = property(asdict)

    @staticmethod
    def from_yaml(path: str = 'configs/composers/standard.yaml') -> ComposerConfig:
        with open(path) as stream:
            return ComposerConfig(**yaml.safe_load(stream))
