from pipeline.data.preprocessing.lm_preprocessor import LMPreprocessor

import torch
from transformers import BatchEncoding


class CompletionLossPreprocessor(LMPreprocessor):
    def get_loss_mask(self,
                      tokenized_completions: BatchEncoding,
                      target_attn_mask: torch.Tensor,
                      ) -> torch.Tensor:
        position_ids = torch.arange(target_attn_mask.shape[-1])
        num_informative_tokens = target_attn_mask.sum(dim=-1, keepdim=True)
        completions_len = tokenized_completions.length.unsqueeze(-1)
        num_loss_tokens = (self.loss_ratio * completions_len).ceil().long()

        loss_mask = (num_informative_tokens - num_loss_tokens <= position_ids)
        loss_mask.logical_and_(target_attn_mask)
        return loss_mask
