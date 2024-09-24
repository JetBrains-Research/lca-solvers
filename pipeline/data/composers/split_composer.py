from pipeline.data.composed_datapoint import ComposedBlockDatapoint, BatchComposedBlockDatapoint
from pipeline.data.composers.chain import UnsafeComposerChain
from pipeline.data.datapoint import Datapoint, BatchDatapoint

from typing import Any

from transformers import PreTrainedTokenizerBase


class SplitComposer:
    def __init__(self,
                 max_block_size: int,
                 max_num_blocks: int,
                 block_sep: str,
                 recalculate_random_category: bool,
                 tokenizer: PreTrainedTokenizerBase,
                 composer_chain: UnsafeComposerChain,
                 ) -> None:
        self.max_block_size = max_block_size - 1  # exclude BOS
        self.max_num_blocks = max_num_blocks - 1  # exclude completion block
        self.block_sep = block_sep
        self.recalculate_random_category = recalculate_random_category
        self.tokenizer = tokenizer
        self.composer_chain = composer_chain

        self.max_token_len = max(map(len, tokenizer.get_vocab()))
        self.min_token_len = min(map(len, tokenizer.get_vocab()))

    def compose_context_blocks(self, datapoint: Datapoint) -> list[str]:
        raw_blocks = self.composer_chain(datapoint)[::-1]
        composed_blocks = list()

        for block in raw_blocks:
            content = block.content.rstrip(self.block_sep[:1]) + self.block_sep

            if len(content) * self.min_token_len <= self.max_block_size:
                composed_blocks.append(content)

            elif len(content) * self.max_token_len <= self.max_block_size:
                num_tokens = len(self.tokenizer(content, return_attention_mask=False).input_ids)
                if num_tokens <= self.max_block_size:
                    composed_blocks.append(content)

            if len(composed_blocks) == self.max_num_blocks:
                break

        return composed_blocks[::-1]

    @staticmethod
    def compose_completion_block(datapoint: Datapoint) -> str:
        completion = datapoint.completion_file['content']
        if not completion.endswith('\n'):
            completion += '\n'  # instead of EOS token

        return completion

    def compose(self, datapoint: dict[str, Any]) -> ComposedBlockDatapoint:
        datapoint = Datapoint(**datapoint)

        if self.recalculate_random_category:
            datapoint.recalculate_random_category()

        return ComposedBlockDatapoint(
            context_blocks=self.compose_context_blocks(datapoint),
            completion_block=self.compose_completion_block(datapoint),
            completion_lines=datapoint.completion_lines,
        )

    def compose_batch(self, batch: BatchDatapoint) -> BatchComposedBlockDatapoint:
        if len(next(iter(batch.values()))) != 1:
            raise ValueError('This composer only accepts batch_size = 1.')

        batch = self.compose({k: v[0] for k, v in batch.items()})
        batch = {k: [v] for k, v in batch.items()}
        return batch
