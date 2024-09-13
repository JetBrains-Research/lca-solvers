from typing import Iterator

import torch
import torch.nn as nn


class GeneratorMixin:
    def __init__(self, model: nn.Module, hs_input: bool) -> None:
        self.model = model
        self.hs_input = hs_input

    @torch.inference_mode
    def produce_logits(self,
                       head_blocks: list[torch.Tensor],
                       tail_blocks: list[torch.Tensor],
                       target_length: int,
                       ) -> Iterator[torch.Tensor]:
        for start_idx in range(len(head_blocks) - 1, -1, -1):
            model_input = torch.concat([head_blocks[start_idx]] + tail_blocks[start_idx:])
            model_input = model_input.unsqueeze(0)  # batch dimension
            model_input = {('input_embeds' if self.hs_input else 'input_ids'): model_input}

            try:
                logits = self.model(**model_input).logits
            except torch.cuda.OutOfMemoryError:
                break

            logits = logits.squeeze(0)[-target_length:]
            yield logits
