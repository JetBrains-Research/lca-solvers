from pipeline.configs.config_base import Config

from dataclasses import dataclass


@dataclass
class LMPreprocessorConfig(Config):
    max_seq_len: int
    context_tokens: int | float
    loss_ratio: float
    n_chars_per_token: int = 4
    verbose: int = 1
