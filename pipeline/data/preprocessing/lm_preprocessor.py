from pipeline.data.categories import CATEGORY2ID, UNDEFINED_CATEGORY_ID
from pipeline.data.composed_datapoint import BatchComposedDatapoint
from pipeline.data.datapoint import CompletionLines
from pipeline.data.preprocessing.preprocessor_base import PreprocessedBatch, PreprocessorBase

import math
import re
import warnings

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


class LMPreprocessor(PreprocessorBase):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_seq_len: int,
                 context_tokens: int | float,
                 loss_ratio: float,
                 num_chars_per_token: int,
                 verbose: int,
                 ) -> None:
        if not 0 < loss_ratio <= 1:
            raise ValueError('loss_ratio must be selected from the interval (0, 1]. '
                             f'Got {loss_ratio} instead.')

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len + 1  # for targets

        if isinstance(context_tokens, float):
            if not 0 <= context_tokens <= 1:
                raise ValueError('The context ratio must be between 0 and 1.')

            context_tokens = int(max_seq_len * context_tokens)
        self.context_tokens = context_tokens

        self.loss_ratio = loss_ratio  # can be used in the future in get_loss_mask
        self._loss_mask = torch.zeros(1, max_seq_len, dtype=torch.bool)
        self._loss_mask[:, -math.ceil(loss_ratio * max_seq_len):] = 1

        self.num_chars_per_token = num_chars_per_token
        self.verbose = verbose

    def _inc_num_chars_per_token(self) -> None:
        old_value = self.num_chars_per_token
        self.num_chars_per_token *= 2

        if self.verbose >= 1:
            warnings.warn(
                f'num_chars_per_token has been increased from {old_value} to {self.num_chars_per_token} '
                'due to an underestimation of the length of the truncated character sequence.')

    def tokenize_pre_context_prompt(self, prompts: list[str]) -> BatchEncoding:
        trunc_upper_bound = self.max_seq_len - self.context_tokens
        char_trunc_upper_bound = self.num_chars_per_token * trunc_upper_bound

        tokenized_prompts = self.tokenizer(
            text=[prompt[-char_trunc_upper_bound:] for prompt in prompts],
            add_special_tokens=True,  # bos
            return_attention_mask=False,
            return_length=True,
        )

        tokenized_prompts.length = torch.tensor(tokenized_prompts.length)
        overflow_chars = torch.tensor([len(prompt) > char_trunc_upper_bound for prompt in prompts])
        underflow_tokens = (tokenized_prompts.length < trunc_upper_bound)
        overflow_tokens = (tokenized_prompts.length > trunc_upper_bound)

        if torch.any(overflow_chars & underflow_tokens):
            self._inc_num_chars_per_token()
            return self.tokenize_pre_context_prompt(prompts)

        for i in range(len(prompts)):
            if overflow_tokens[i]:
                tokenized_prompts.input_ids[i] = (
                        [self.tokenizer.bos_token_id] +
                        tokenized_prompts.input_ids[i][-(trunc_upper_bound - 1):]
                )

                if self.verbose >= 2:
                    warnings.warn(f'The pre context prompt at position number {i} in the given '
                                  'batch has been truncated due to its excessive length.')
        tokenized_prompts.length.clip_(max=trunc_upper_bound)

        return tokenized_prompts

    def tokenize_composed_completion(self, completions: list[str], prompts_len: torch.Tensor) -> BatchEncoding:
        trunc_upper_bound = self.max_seq_len - self.context_tokens - prompts_len
        char_trunc_upper_bound = self.num_chars_per_token * max(trunc_upper_bound)
        trunc_completions = [completion[:char_trunc_upper_bound] for completion in completions]

        tokenized_completions = self.tokenizer(
            text=trunc_completions,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            return_length=True,
        )

        tokenized_completions.length = torch.tensor(tokenized_completions.length)
        overflow_chars = torch.tensor([len(completion) > char_trunc_upper_bound for completion in completions])
        underflow_tokens = (tokenized_completions.length < trunc_upper_bound)
        overflow_tokens = (tokenized_completions.length > trunc_upper_bound)

        if torch.any(overflow_chars & underflow_tokens):
            self._inc_num_chars_per_token()
            return self.tokenize_composed_completion(completions, prompts_len)

        tokenized_completions['newline_positions'] = [
            [match.start() for match in re.finditer('\n', completion)]
            for completion in trunc_completions
        ]

        for i in range(len(completions)):
            if overflow_tokens[i]:
                tokenized_completions.input_ids[i] = tokenized_completions.input_ids[i][:trunc_upper_bound[i]]
                tokenized_completions.offset_mapping[i] = tokenized_completions.offset_mapping[i][:trunc_upper_bound[i]]
        tokenized_completions.length.clip_(max=trunc_upper_bound)

        return tokenized_completions

    def tokenize_composed_context(self,
                                  contexts: list[str],
                                  prompts_len: torch.Tensor,
                                  completions_len: torch.Tensor,
                                  ) -> BatchEncoding:
        contexts_len = self.max_seq_len - prompts_len - completions_len
        char_trunc_upper_bound = self.num_chars_per_token * max(contexts_len)

        tokenized_contexts = self.tokenizer(
            text=[ctx[-char_trunc_upper_bound:] for ctx in contexts],
            add_special_tokens=False,
            return_attention_mask=False,
            return_length=True,
        )

        tokenized_contexts.length = torch.tensor(tokenized_contexts.length)
        overflow_chars = torch.tensor([len(ctx) > char_trunc_upper_bound for ctx in contexts])
        underflow_tokens = (tokenized_contexts.length < contexts_len)
        overflow_tokens = (tokenized_contexts.length > contexts_len)

        if torch.any(overflow_chars & underflow_tokens):
            self._inc_num_chars_per_token()
            return self.tokenize_composed_context(contexts, prompts_len, completions_len)

        if torch.any(~overflow_chars & underflow_tokens):
            raise ValueError('Not enough data to satisfy context_tokens.')

        for i in range(len(contexts)):
            if overflow_tokens[i]:
                tokenized_contexts.input_ids[i] = tokenized_contexts.input_ids[i][-contexts_len[i]:]
        tokenized_contexts.length = contexts_len

        return tokenized_contexts

    def get_loss_mask(self, batch_size: int) -> torch.Tensor:
        return self._loss_mask.expand(batch_size, -1)

    def get_category_ids(self,
                         completion_lines: list[CompletionLines],
                         tokenized_completions: BatchEncoding,
                         batch_size: int,
                         ) -> torch.Tensor:
        category_ids = torch.full((batch_size, self.max_seq_len - 1), UNDEFINED_CATEGORY_ID)

        for sample_idx in range(batch_size):
            t_completion_start = self.max_seq_len - tokenized_completions.length[sample_idx] - 1
            newline_positions = tokenized_completions.newline_positions[sample_idx]
            offset_mapping = tokenized_completions.offset_mapping[sample_idx]

            newline_positions.append(float('inf'))
            line2category = {
                line_idx: CATEGORY2ID[category]
                for category, line_category_ids in completion_lines[sample_idx].items()
                for line_idx in line_category_ids
            }

            line_idx = 0
            category_id = line2category.get(line_idx)

            for token_idx, (char_start, _) in enumerate(offset_mapping, start=t_completion_start):
                if char_start > newline_positions[line_idx]:
                    line_idx += 1
                    category_id = line2category.get(line_idx)

                if category_id is not None:
                    category_ids[sample_idx, token_idx] = category_id

        return category_ids

    def __call__(self, batch: BatchComposedDatapoint) -> PreprocessedBatch:
        tokenized_prompts = self.tokenize_pre_context_prompt(
            batch['pre_context_prompt'],
        )
        tokenized_completions = self.tokenize_composed_completion(
            batch['composed_completion'],
            tokenized_prompts.length,
        )
        tokenized_contexts = self.tokenize_composed_context(
            batch['composed_context'],
            tokenized_prompts.length,
            tokenized_completions.length,
        )

        batch_size = len(tokenized_completions.length)
        tokenized_batch = torch.tensor([sum(p_c_c, []) for p_c_c in zip(
            tokenized_prompts.input_ids,
            tokenized_contexts.input_ids,
            tokenized_completions.input_ids
        )])

        return PreprocessedBatch(
            input_ids=tokenized_batch[:, :-1],
            target_ids=tokenized_batch[:, 1:],
            loss_mask=self.get_loss_mask(batch_size),
            category_ids=self.get_category_ids(
                batch['completion_lines'], tokenized_completions, batch_size),
        )
