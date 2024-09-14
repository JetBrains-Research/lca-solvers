from cache_study.model import split_model
from composers.chain.chain import Chunk, UnsafeComposerChain
from composers.data.datapoint import Datapoint

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, Iterator, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


class BOSUsage(str, Enum):
    DISABLED = 'disabled'
    IN_EACH_BLOCK = 'in_each_block'
    HEAD_BLOCKS_ONLY = 'head_blocks_only'
    TAIL_SEAM = 'tail_seam'


@dataclass
class PipelineOutput:
    OOM: Literal['no', 'encoder', 'generator'] = 'no'
    num_tokens: int = 0
    block_names: list[str] = field(default_factory=list)
    cross_entropy: list[float] = field(default_factory=list)


class Pipeline:
    def __init__(self,
                 composer: UnsafeComposerChain,
                 tokenizer: PreTrainedTokenizerBase,
                 bos_usage: BOSUsage,
                 full_model: nn.Module,
                 num_gen_layers: int | None,
                 ) -> None:
        if bos_usage == BOSUsage.TAIL_SEAM and num_gen_layers is None:
            raise ValueError('Tail seaming is only possible for TrialPipeline.')

        self.composer = composer
        self.tokenizer = tokenizer
        self.bos_usage = bos_usage
        self.encoder, self.generator = split_model(full_model, num_gen_layers)

    @property
    def device(self) -> torch.device:
        return self.generator.device

    def preprocess(self,
                   datapoint: dict[str, Any],
                   batch_size: int = 128,
                   max_num_blocks: int | None = None,
                   ) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        datapoint = Datapoint(**datapoint)

        block_names = [datapoint.completion_file['filename']]
        block_sequence = [datapoint.completion_file['content']]

        for block in self.composer(datapoint)[::-1]:
            file_ref = block.file_ref if isinstance(block, Chunk) else block
            block_names.append(file_ref.metadata['filename'])
            block_sequence.append(block.content.rstrip('\n') + '\n\n')

        if max_num_blocks is not None:
            block_sequence = block_sequence[:max_num_blocks]

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
                block = torch.tensor(block, device=self.device)
                head_blocks.append(block[head_block_start:])
                tail_blocks.append(block[tail_block_start:])

        target_ids = head_blocks[0][1:]
        head_blocks[0] = head_blocks[0][:-1]
        tail_blocks[0] = tail_blocks[0][:-1]
        tail_blocks = [torch.tensor([], dtype=torch.long, device=self.device)] + tail_blocks[:-1]

        return block_names, head_blocks, tail_blocks, target_ids

    def encode(self,
               head_blocks: list[torch.Tensor],
               tail_blocks: list[torch.Tensor],
               ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        prev_head_block = None

        for head_block, tail_block in zip(head_blocks, tail_blocks):
            if self.bos_usage == BOSUsage.TAIL_SEAM:
                assert head_block[0] == 32013  # TODO: remove me later

            try:
                head_block = head_block.unsqueeze(0)
                head_block = self.encoder(head_block).last_hidden_state
                head_block = head_block.squeeze(0)

                if tail_block.numel():
                    tail_block = tail_block.unsqueeze(0)
                    tail_block = prev_head_block or self.encoder(tail_block).last_hidden_state
                    tail_block = tail_block.squeeze(0)

            except torch.cuda.OutOfMemoryError:
                yield None, None
                break

            if self.bos_usage == BOSUsage.TAIL_SEAM:
                tail_block = tail_block[1:]  # remove hidden states vector corresponding to the BOS

            if self.bos_usage != BOSUsage.HEAD_BLOCKS_ONLY:
                prev_head_block = head_block

            yield head_block, tail_block

    def produce_logits(self) -> Generator[torch.Tensor, tuple[torch.Tensor, torch.Tensor], None]:
        dtype = torch.long if self.encoder is None else self.generator.dtype
        cumulative_input = torch.tensor([], dtype=dtype, device=self.device)
        head_block, tail_block = yield

        while True:
            cumulative_input = torch.concat([tail_block, cumulative_input])

            model_input = torch.concat([head_block, cumulative_input])
            model_input = model_input.unsqueeze(0)  # batch dimension
            model_input = {('input_ids' if self.encoder is None else 'inputs_embeds'): model_input}

            try:
                head_block, tail_block = yield self.generator(**model_input).logits.squeeze(0)
            except torch.cuda.OutOfMemoryError:
                yield None
                break

    @torch.inference_mode
    def __call__(self, datapoint: dict[str, Any]) -> PipelineOutput:
        output = PipelineOutput()

        block_names, *blocks, target_ids = self.preprocess(datapoint)
        blocks = (zip if self.encoder is None else self.encode)(*blocks)
        logits_generator = self.produce_logits(); next(logits_generator)

        for block_name, (head_block, tail_block) in zip(block_names, blocks):
            torch.cuda.empty_cache()

            if head_block is None:
                output.OOM = 'encoder'
                return output

            logits = logits_generator.send((head_block, tail_block))

            if logits is None:
                output.OOM = 'generator'
                return output

            output.num_tokens = logits.shape[0]
            logits = logits[-len(target_ids):]
            output.block_names.append(block_name)
            loss = F.cross_entropy(logits, target_ids).item()
            output.cross_entropy.append(loss)

        return output
