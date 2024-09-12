from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass
class PreprocessorConfig(ConfigBase):
    tokenizer: PreTrainedTokenizerBase
    max_seq_len: int
    context_tokens: int | float
    loss_ratio: float
    num_chars_per_token: int
    use_sep_token: bool
    padding: bool
    verbose: bool = True
