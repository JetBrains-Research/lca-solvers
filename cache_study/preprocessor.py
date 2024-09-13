from composers.chain.chain import UnsafeComposerChain
from composers.data.datapoint import Datapoint

from enum import Enum
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


class BOSUsage(str, Enum):
    DISABLED = 'disabled'
    IN_EACH_BLOCK = 'in_each_block'
    HEAD_BLOCKS_ONLY = 'head_blocks_only'
    TAIL_SEAM = 'tail_seam'


class PreprocessorMixin:
    def __init__(self,
                 composer: UnsafeComposerChain,
                 tokenizer: PreTrainedTokenizerBase,
                 bos_usage: BOSUsage,
                 ) -> None:
        self.composer = composer
        self.tokenizer = tokenizer
        self.bos_usage = bos_usage

    def preprocess(self,
                   datapoint: dict[str, Any],
                   device: torch.device,
                   batch_size: int = 4,  # TODO: tune
                   max_num_blocks: int | None = None,
                   ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        datapoint = Datapoint(**datapoint)

        block_sequence = [block.content.rstrip('\n') + '\n\n' for block in self.composer(datapoint)]
        block_sequence.append(datapoint.completion_file['content'])
        if max_num_blocks is not None:
            block_sequence = block_sequence[-max_num_blocks:]

        head_blocks = list()
        tail_blocks = list()
        head_block_start = int(self.bos_usage == BOSUsage.DISABLED)
        tail_block_start = head_block_start + int(self.bos_usage == BOSUsage.HEAD_BLOCKS_ONLY)

        for batch_start in range(0, len(block_sequence), batch_size):
            tokenized_batch = self.tokenizer(
                text=block_sequence[batch_start:(batch_start + batch_size)],
                return_attention_mask=False,
            ).input_ids

            for block in tokenized_batch:
                block = torch.tensor(block, device=device)
                head_blocks.append(block[head_block_start:])
                tail_blocks.append(block[tail_block_start:])

        target_ids = head_blocks[-1][1:]
        head_blocks[-1] = head_blocks[-1][:-1]
        tail_blocks[-1] = tail_blocks[-1][:-1]
        tail_blocks = tail_blocks[1:]

        return head_blocks, tail_blocks, target_ids
