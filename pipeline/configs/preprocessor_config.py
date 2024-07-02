from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass
class LMPreprocessorConfig:
    tokenizer: PreTrainedTokenizerBase
    max_seq_len: int
    context_tokens: int | float
    loss_ratio: float
    n_chars_per_token: int = 4
    verbose: int = 1
