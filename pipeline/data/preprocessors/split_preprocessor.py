from pipeline.data.composed_datapoint import BatchComposedBlockDatapoint
from pipeline.data.categories import CATEGORY2ID, UNDEFINED_CATEGORY_ID
from pipeline.data.datapoint import CompletionLines
from pipeline.data.preprocessors.preprocessor_base import PreprocessedBatch, AmortizedPreprocessorBase

import math
import re

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


class SplitPreprocessor(AmortizedPreprocessorBase):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_completion_len: int,
                 loss_ratio: float,
                 num_chars_per_token: int,
                 verbose: bool,
                 ) -> None:
        super().__init__(num_chars_per_token, verbose)

        if not 0 < loss_ratio <= 1:
            raise ValueError('loss_ratio must be selected from the interval (0, 1]. '
                             f'Got {loss_ratio} instead.')

        tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True
        tokenizer.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum'] = True

        self.tokenizer = tokenizer
        self.max_completion_len = max_completion_len
        self.loss_ratio = loss_ratio

    def tokenize_composed_completion(self, completion_block: str) -> BatchEncoding:
        char_trunc_upper_bound = self.num_chars_per_token * self.max_completion_len
        trunc_completion = completion_block[:char_trunc_upper_bound]

        tokenized_completion = self.tokenizer(
            text=trunc_completion,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )

        overflow_chars = (len(completion_block) > char_trunc_upper_bound)
        underflow_tokens = (len(tokenized_completion.input_ids) < self.max_completion_len)

        if overflow_chars and underflow_tokens:
            self._inc_num_chars_per_token()
            return self.tokenize_composed_completion(completion_block)

        tokenized_completion.input_ids = tokenized_completion.input_ids[:self.max_completion_len]
        tokenized_completion.offset_mapping = tokenized_completion.offset_mapping[:self.max_completion_len]
        tokenized_completion['newline_positions'] = [
            match.start() for match in re.finditer('\n', trunc_completion)
        ]

        return tokenized_completion

    def tokenize_composed_context(self, context_blocks: list[str]) -> BatchEncoding:
        return self.tokenizer(
            text=context_blocks,
            add_special_tokens=False,
            return_attention_mask=False,
        )

    def get_loss_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[:, -math.ceil(self.loss_ratio * seq_len):] = True
        return mask

    def get_completion_mask(self, seq_len: int, completion_len: int) -> torch.Tensor:
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[:, -completion_len:] = True
        return mask

    @staticmethod
    def get_category_ids(seq_len: int,
                         completion_len: int,
                         newline_positions: list[int],
                         offset_mapping: list[tuple[int, int]],
                         completion_lines: CompletionLines,
                         ) -> torch.Tensor:
        category_ids = torch.full((seq_len,), UNDEFINED_CATEGORY_ID, dtype=torch.long)
        t_completion_start = seq_len - completion_len

        newline_positions.append(float('inf'))
        line2category = {
            line_idx: CATEGORY2ID[category]
            for category, line_category_ids in completion_lines.items()
            for line_idx in line_category_ids
        }

        line_idx = 0
        category_id = line2category.get(line_idx)

        for token_idx, (char_start, _) in enumerate(offset_mapping, start=t_completion_start):
            if char_start > newline_positions[line_idx]:
                line_idx += 1
                category_id = line2category.get(line_idx)

            if category_id is not None:
                category_ids[token_idx] = category_id

        category_ids = category_ids.unsqueeze(0)
        return category_ids

    def __call__(self, batch: BatchComposedBlockDatapoint) -> PreprocessedBatch:
        if len(next(iter(batch.values()))) != 1:
            raise ValueError('This preprocessor only accepts batch_size = 1.')

        tokenized_completion = self.tokenize_composed_completion(batch['completion_block'][0])
        tokenized_context = self.tokenize_composed_context(batch['context_blocks'][0])

        tokenized_batch = tokenized_context.input_ids + [tokenized_completion.input_ids]
        tokenized_batch = [[self.tokenizer.bos_token_id] + block for block in tokenized_batch]

        padded_batch = self.tokenizer.pad(
            encoded_inputs={'input_ids': tokenized_batch},
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt',
        )

        attention_mask = padded_batch.attention_mask.bool()
        last_token_idx = completion_len = attention_mask[-1].sum() - 1

        attention_mask[1:, 0] = False  # remove BOS tokens except for the first one
        target_ids = padded_batch.input_ids[attention_mask][None, 1:]
        seq_len = target_ids.shape[-1]

        input_ids = padded_batch.input_ids.unsqueeze(0)
        input_ids[:, -1, last_token_idx] = self.tokenizer.pad_token_id

        input_attn_mask = padded_batch.attention_mask.unsqueeze(0)
        input_attn_mask[:, -1, last_token_idx] = 0

        return PreprocessedBatch(
            input_ids=input_ids,
            target_ids=target_ids,
            loss_mask=self.get_loss_mask(seq_len),
            completion_mask=self.get_completion_mask(seq_len, completion_len),
            category_ids=self.get_category_ids(
                seq_len=seq_len,
                completion_len=completion_len,
                newline_positions=tokenized_completion.newline_positions,
                offset_mapping=tokenized_completion.offset_mapping,
                completion_lines=batch['completion_lines'][0],
            ),
            input_attn_mask=input_attn_mask,
            target_attn_mask=torch.ones_like(target_ids, dtype=torch.bool),
        )
