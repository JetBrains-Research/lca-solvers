from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import PREPROCESSOR_YAML

from dataclasses import dataclass


@dataclass
class LMPreprocessorConfig(ConfigBase):
    _default_path = PREPROCESSOR_YAML

    max_seq_len: int
    context_tokens: int | float
    loss_ratio: float
    num_chars_per_token: int
    verbose: int = 1
