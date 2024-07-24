from pipeline.data.preprocessing.lm_preprocessor import LMPreprocessor

import torch
from transformers import BatchEncoding


class CompletionLossPreprocessor(LMPreprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        delattr(self, '_loss_mask')

    def get_loss_mask(self, tokenized_completions: BatchEncoding) -> torch.Tensor:
        neg_position_ids = torch.arange(-self.max_seq_len, 0)
        completions_len = tokenized_completions.length.unsqueeze(-1)
        num_loss_tokens = (self.loss_ratio * completions_len).ceil().long()
        loss_mask = (-completions_len <= neg_position_ids) & (neg_position_ids < -completions_len + num_loss_tokens)
        return loss_mask
