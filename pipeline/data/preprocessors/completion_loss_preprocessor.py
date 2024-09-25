from pipeline.data.categories import CATEGORY2ID, UNDEFINED_CATEGORY_ID
from pipeline.data.composed_datapoint import BatchComposedDatapoint
from pipeline.data.datapoint import CompletionLines
from pipeline.data.preprocessors.preprocessor_base import PreprocessedBatch, AmortizedPreprocessorBase

import re
import warnings

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


class CompletionLossPreprocessor(AmortizedPreprocessorBase):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_seq_len: int,
                 context_tokens: int | float,
                 loss_ratio: float,
                 num_chars_per_token: int,
                 use_sep_token: bool,  # appended to the context
                 padding: bool,
                 verbose: bool,
                 ) -> None:
        super().__init__(num_chars_per_token, verbose)

        if not 0 < loss_ratio <= 1:
            raise ValueError('loss_ratio must be selected from the interval (0, 1]. '
                             f'Got {loss_ratio} instead.')

        if padding:
            tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True
        tokenizer.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum'] = True

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if isinstance(context_tokens, float):
            if not 0 <= context_tokens <= 1:
                raise ValueError('The context ratio must be between 0 and 1.')

            context_tokens = int(max_seq_len * context_tokens)
        self.context_tokens = context_tokens

        self.loss_ratio = loss_ratio
        self.use_sep_token = use_sep_token
        self.padding = padding

    def tokenize_pre_context_prompt(self, prompts: list[str]) -> BatchEncoding:
        char_trunc_upper_bound = self.num_chars_per_token * self.max_seq_len

        tokenized_prompts = self.tokenizer(
            text=[prompt[-char_trunc_upper_bound:] for prompt in prompts],
            add_special_tokens=False,
            return_attention_mask=False,
        )

        for tokenized_prompt, prompt in zip(tokenized_prompts.input_ids, prompts):
            overflow_chars = len(prompt) > char_trunc_upper_bound
            underflow_tokens = len(tokenized_prompt) < self.max_seq_len

            if overflow_chars and underflow_tokens:
                self._inc_num_chars_per_token()
                return self.tokenize_pre_context_prompt(prompts)

        return tokenized_prompts

    def tokenize_composed_completion(self, completions: list[str]) -> BatchEncoding:
        char_trunc_upper_bound = self.num_chars_per_token * self.max_seq_len
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
        underflow_tokens = (tokenized_completions.length < self.max_seq_len)

        if torch.any(overflow_chars & underflow_tokens):
            self._inc_num_chars_per_token()
            return self.tokenize_composed_completion(completions)

        tokenized_completions['newline_positions'] = [
            [match.start() for match in re.finditer('\n', completion)]
            for completion in trunc_completions
        ]

        return tokenized_completions

    def tokenize_composed_context(self, contexts: list[str]) -> BatchEncoding:
        char_trunc_upper_bound = self.num_chars_per_token * self.max_seq_len

        tokenized_contexts = self.tokenizer(
            text=[ctx[-char_trunc_upper_bound:] for ctx in contexts],
            add_special_tokens=False,
            return_attention_mask=False,
        )

        for tokenized_ctx, ctx in zip(tokenized_contexts.input_ids, contexts):
            overflow_chars = len(ctx) > char_trunc_upper_bound
            underflow_tokens = len(tokenized_ctx) < self.max_seq_len

            if overflow_chars and underflow_tokens:
                self._inc_num_chars_per_token()
                return self.tokenize_composed_context(contexts)

            if not overflow_chars and len(tokenized_ctx) < self.context_tokens:
                if not self.padding:
                    raise ValueError('Not enough data to satisfy context_tokens.')
                elif self.verbose:
                    warnings.warn('Not enough data to satisfy context_tokens.')

        return tokenized_contexts

    def calc_lens(self,
                  prompt: torch.Tensor,
                  context: torch.Tensor,
                  completion: torch.Tensor,
                  ) -> tuple[int, int, int]:
        if len(context) >= self.context_tokens:
            prompt_len = min(len(prompt), self.max_seq_len - self.context_tokens)
            completion_len = min(len(completion), self.max_seq_len - self.context_tokens - prompt_len)
            context_len = self.max_seq_len - prompt_len - completion_len
        else:
            context_len = len(context)
            prompt_len = min(len(prompt), self.max_seq_len - context_len)
            completion_len = self.max_seq_len - prompt_len - context_len

        return prompt_len, context_len, completion_len

    @staticmethod
    def _get_partial_completion_mask(tokenized_completions: BatchEncoding,
                                     target_attn_mask: torch.Tensor,
                                     ratio: float,
                                     ) -> torch.Tensor:
        position_ids = torch.arange(target_attn_mask.shape[-1])
        num_informative_tokens = target_attn_mask.sum(dim=-1, keepdim=True)
        completions_len = tokenized_completions.length.unsqueeze(-1)
        num_masked_tokens = (ratio * completions_len).ceil().long()
        mask = (num_informative_tokens - num_masked_tokens <= position_ids)
        return mask.logical_and(target_attn_mask)

    def get_loss_mask(self, *args, **kwargs) -> torch.Tensor:
        return self._get_partial_completion_mask(*args, **kwargs, ratio=self.loss_ratio)

    def get_completion_mask(self, *args, **kwargs) -> torch.Tensor:
        return self._get_partial_completion_mask(*args, **kwargs, ratio=1)

    @staticmethod
    def get_category_ids(tokenized_completions: BatchEncoding,
                         completion_lines: list[CompletionLines],
                         target_attn_mask: torch.Tensor,
                         ) -> torch.Tensor:
        category_ids = torch.full_like(target_attn_mask, UNDEFINED_CATEGORY_ID)
        t_completion_start = (target_attn_mask.sum(dim=-1) - tokenized_completions.length).tolist()
        batch_size = len(tokenized_completions.length)

        for sample_idx in range(batch_size):
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

            for token_idx, (char_start, _) in enumerate(offset_mapping, start=t_completion_start[sample_idx]):
                if char_start > newline_positions[line_idx]:
                    line_idx += 1
                    category_id = line2category.get(line_idx)

                if category_id is not None:
                    category_ids[sample_idx, token_idx] = category_id

        return category_ids

    def __call__(self, batch: BatchComposedDatapoint) -> PreprocessedBatch:
        tokenized_prompts = self.tokenize_pre_context_prompt(batch['pre_context_prompt'])
        tokenized_completions = self.tokenize_composed_completion(batch['composed_completion'])
        tokenized_contexts = self.tokenize_composed_context(batch['composed_context'])

        tokenized_batch = list()
        batch_size = len(tokenized_completions.length)

        for sample_idx in range(batch_size):
            prompt = tokenized_prompts.input_ids[sample_idx]
            context = tokenized_contexts.input_ids[sample_idx]
            completion = tokenized_completions.input_ids[sample_idx]

            prompt_len, context_len, completion_len = self.calc_lens(prompt, context, completion)

            prompt = [self.tokenizer.bos_token_id] + prompt[-prompt_len:]
            context = context[-context_len:]
            if self.use_sep_token and context:
                context = context[1:] + [self.tokenizer.sep_token_id]
            completion = completion[:completion_len]

            tokenized_completions.offset_mapping[sample_idx] = \
                tokenized_completions.offset_mapping[sample_idx][:completion_len]
            tokenized_completions.length[sample_idx] = len(completion)

            tokenized_batch.append(prompt + context + completion)

        padded_batch = self.tokenizer.pad(
            encoded_inputs={'input_ids': tokenized_batch},
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt')
        input_attn_mask = padded_batch.attention_mask[:, :-1]
        target_attn_mask = padded_batch.attention_mask[:, 1:]

        return PreprocessedBatch(
            input_ids=padded_batch.input_ids[:, :-1],
            target_ids=padded_batch.input_ids[:, 1:],
            loss_mask=self.get_loss_mask(tokenized_completions, target_attn_mask),
            completion_mask=self.get_completion_mask(tokenized_completions, target_attn_mask),
            category_ids=self.get_category_ids(tokenized_completions, batch['completion_lines'], target_attn_mask),
            input_attn_mask=input_attn_mask,
            target_attn_mask=target_attn_mask.bool(),
        )
