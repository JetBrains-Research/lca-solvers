from pipeline.data.preprocessors.completion_loss_preprocessor import CompletionLossPreprocessor

import torch
from transformers import BatchEncoding


class LMPreprocessor(CompletionLossPreprocessor):
    def get_loss_mask(self,
                      _tokenized_completions: BatchEncoding,
                      target_attn_mask: torch.Tensor,
                      **_kwargs,
                      ) -> torch.Tensor:
        position_ids = torch.arange(target_attn_mask.shape[-1])
        num_informative_tokens = target_attn_mask.sum(dim=-1, keepdim=True)
        num_loss_tokens = (self.loss_ratio * num_informative_tokens).ceil().long()
        loss_mask = (num_informative_tokens - num_loss_tokens <= position_ids)
        return loss_mask.logical_and(target_attn_mask)
