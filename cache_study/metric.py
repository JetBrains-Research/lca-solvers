from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class Metrics:
    cross_entropy: list[float]
    exact_match: list[float]


def exact_match(logits: torch.Tensor, target_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> float:
    pass  # TODO
