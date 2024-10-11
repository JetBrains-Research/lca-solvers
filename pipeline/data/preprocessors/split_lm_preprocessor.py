from pipeline.data.preprocessors.split_completion_loss_preprocessor import SplitCompletionLossPreprocessor

import math

import torch


class SplitLMPreprocessor(SplitCompletionLossPreprocessor):
    def get_loss_mask(self, seq_len: int, *_args, **_kwargs) -> torch.Tensor:
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[:, -math.ceil(self.loss_ratio * seq_len):] = True
        return mask
