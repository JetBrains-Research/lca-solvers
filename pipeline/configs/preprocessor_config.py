from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class LMPreprocessorConfig(ConfigBase):
    max_seq_len: int
    context_tokens: int | float
    loss_ratio: float
    n_chars_per_token: int = 4
    verbose: int = 1
