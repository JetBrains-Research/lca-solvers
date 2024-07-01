from pipeline.data.composed_datapoint import BatchedComposedDatapoint
from pipeline.data.preprocessing.preprocessor_base import PreprocessorBase

import math
import warnings
from typing import TypedDict

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


class LMBatch(TypedDict):
    input_ids: torch.Tensor
    loss_mask: torch.Tensor


# TODO: test
# TODO: reduce repetitive code
class LMPreprocessor(PreprocessorBase):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_seq_len: int,
                 context_tokens: int | float,
                 loss_ratio: float,
                 n_chars_per_token: int = 4,
                 verbose: int = 1,
                 ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if isinstance(context_tokens, float):
            if not 0 <= context_tokens <= 1:
                raise ValueError('The context ratio must be between 0 and 1.')

            context_tokens = int(max_seq_len * context_tokens)
        self.context_tokens = context_tokens

        self.loss_ratio = loss_ratio
        self._loss_mask = torch.zeros(1, max_seq_len, dtype=torch.long)
        self._loss_mask[:, -math.ceil(loss_ratio * max_seq_len):] = 1

        self.n_chars_per_token = n_chars_per_token
        self.verbose = verbose

    def _inc_n_chars_per_token(self, traceback_msg: str = '') -> None:
        old_value = self.n_chars_per_token
        self.n_chars_per_token *= 2

        if self.verbose >= 1:
            warnings.warn(traceback_msg +
                          f'n_chars_per_token has been increased from {old_value} to {self.n_chars_per_token} '
                          'due to an underestimation of the length of the truncated character sequence.')

    def get_loss_mask(self, batch_size: int = 1) -> torch.Tensor:
        # can be easily overridden by subclasses
        return self._loss_mask.expand(batch_size, -1)

    def tokenize_pre_context_prompt(self, prompts: list[str]) -> BatchEncoding:
        trunc_upper_bound = self.max_seq_len - self.context_tokens
        char_trunc_upper_bound = self.n_chars_per_token * trunc_upper_bound

        tokenized_prompts = self.tokenizer(  # TODO: refactor args
            text=[prompt[-char_trunc_upper_bound:] for prompt in prompts],
            add_special_tokens=True,  # bos
            padding=False,
            truncation=False,
            max_length=None,
            return_tensors=None,
            return_attention_mask=False,
            return_length=True,
        )

        tokenized_prompts.length = torch.tensor(tokenized_prompts.length)
        overflow_chars = torch.tensor([len(prompt) > char_trunc_upper_bound for prompt in prompts])
        underflow_tokens = (tokenized_prompts.length < trunc_upper_bound)
        overflow_tokens = (tokenized_prompts.length > trunc_upper_bound)

        if torch.any(overflow_chars & underflow_tokens):
            self._inc_n_chars_per_token(
                traceback_msg='Warning message from LMPreprocessor.tokenize_pre_context_prompt method: ')  # TODO: automatic naming
            return self.tokenize_pre_context_prompt(prompts)

        # truncation
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
        char_trunc_upper_bound = self.n_chars_per_token * max(trunc_upper_bound)

        tokenized_completions = self.tokenizer(
            text=[completion[:char_trunc_upper_bound] for completion in completions],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            max_length=None,
            return_tensors=None,
            return_attention_mask=False,
            return_length=True,
        )

        tokenized_completions.length = torch.tensor(tokenized_completions.length)
        overflow_chars = torch.tensor([len(completion) > char_trunc_upper_bound for completion in completions])
        underflow_tokens = (tokenized_completions.length < trunc_upper_bound)
        overflow_tokens = (tokenized_completions.length > trunc_upper_bound)

        if torch.any(overflow_chars & underflow_tokens):
            self._inc_n_chars_per_token(
                traceback_msg='Warning message from LMPreprocessor.tokenize_composed_completion method: ')  # TODO: automatic naming
            return self.tokenize_composed_completion(completions, prompts_len)

        # truncation
        for i in range(len(completions)):
            if overflow_tokens[i]:
                tokenized_completions.input_ids[i] = tokenized_completions.input_ids[i][:trunc_upper_bound[i]]
        tokenized_completions.length.clip_(max=trunc_upper_bound)

        return tokenized_completions

    def tokenize_composed_context(self,
                                  contexts: list[str],
                                  prompts_len: torch.Tensor,
                                  completions_len: torch.Tensor,
                                  ) -> BatchEncoding:
        contexts_len = self.max_seq_len - prompts_len - completions_len
        char_trunc_upper_bound = self.n_chars_per_token * max(contexts_len)

        tokenized_contexts = self.tokenizer(
            text=[ctx[-char_trunc_upper_bound:] for ctx in contexts],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            max_length=None,
            return_tensors=None,
            return_attention_mask=False,
            return_length=True,
        )

        tokenized_contexts.length = torch.tensor(tokenized_contexts.length)
        overflow_chars = torch.tensor([len(ctx) > char_trunc_upper_bound for ctx in contexts])
        underflow_tokens = (tokenized_contexts.length < contexts_len)
        overflow_tokens = (tokenized_contexts.length > contexts_len)

        if torch.any(overflow_chars & underflow_tokens):
            self._inc_n_chars_per_token(
                traceback_msg='Warning message from LMPreprocessor.tokenize_composed_context method: ')  # TODO: automatic naming
            return self.tokenize_composed_context(contexts, prompts_len, completions_len)

        if torch.any(~overflow_chars & underflow_tokens):
            raise ValueError('Not enough data to satisfy context_tokens.')

        # truncation
        for i in range(len(contexts)):
            if overflow_tokens[i]:
                tokenized_contexts.input_ids[i] = tokenized_contexts.input_ids[i][-contexts_len[i]:]
        tokenized_contexts.length = contexts_len

        return tokenized_contexts

    def __call__(self, batch: BatchedComposedDatapoint) -> LMBatch:
        # TODO: bos tokens must be only in the begging of sample
        # TODO: warn if padding is used

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

        return LMBatch(
            input_ids=torch.tensor([sum(p_c_c, []) for p_c_c in zip(
                tokenized_prompts.input_ids,
                tokenized_contexts.input_ids,
                tokenized_completions.input_ids
            )]),
            loss_mask=self.get_loss_mask(batch_size=len(tokenized_prompts.length)),
        )
