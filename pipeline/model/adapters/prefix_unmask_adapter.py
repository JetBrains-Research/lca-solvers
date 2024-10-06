# see notebooks/attention_benchmarking.ipynb for more details
from pipeline.data.preprocessors.preprocessor_base import BatchMetadata
from pipeline.model.adapters.adapter_base import AdapterBase
from pipeline.model.init import AttentionImplementation

import math
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from einops import repeat
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import LlamaFlashAttention2


class AttentionWeightsForm(str, Enum):
    NO_CAUSAL = 'no_causal'
    CAUSAL = 'causal'
    PREFIX_UNMASK = 'prefix_unmask'
    SCORES_ONLY = 'scores_only'

    def calc_attn_mask(self, seq_len: int, prefix_len: int) -> torch.Tensor:
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        if self == self.NO_CAUSAL:
            return mask

        mask = torch.tril(mask)
        if self == self.CAUSAL:
            return mask

        mask |= (torch.arange(seq_len) < prefix_len)
        return mask


@torch.inference_mode
def calc_attn_weights(query_states: torch.Tensor,
                      key_states: torch.Tensor,
                      prefix_len: int,
                      form: AttentionWeightsForm,
                      ) -> torch.Tensor:
    query_states = query_states.float()
    key_states = key_states.float()
    key_states = repeat(key_states, 'b s h d -> b s (h g) d', g=query_states.shape[2] // key_states.shape[2])
    d = query_states.shape[-1]

    scores = torch.einsum('bthd,bshd->bhts', query_states / math.sqrt(d), key_states)
    if form == AttentionWeightsForm.SCORES_ONLY:
        return scores

    attention_mask = form.calc_attn_mask(query_states.shape[1], prefix_len).to(scores.device)
    attention_bias = torch.zeros_like(scores)
    attention_bias.masked_fill_(~attention_mask, -torch.inf)

    attention_weights = torch.softmax(scores + attention_bias, dim=-1)
    return attention_weights


class PrefixUnmaskAttention(LlamaFlashAttention2):
    prefix_len = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_attentions = False
        self.attn_weights = None
        # does not affect the calculations, only the returned weight type
        self.attention_weights_form = AttentionWeightsForm.PREFIX_UNMASK

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        self.output_attentions = args[4] if len(args) >= 4 else kwargs.get('output_attentions', False)
        attn_output, _, past_key_value = super().forward(*args, **kwargs)
        return attn_output, self.attn_weights, past_key_value

    def _flash_attention_forward(self,
                                 query_states: torch.Tensor,
                                 key_states: torch.Tensor,
                                 value_states: torch.Tensor,
                                 attention_mask: torch.Tensor | None,
                                 _query_length: int,
                                 dropout: float = 0,
                                 _softmax_scale: float | None = None,
                                 ) -> torch.Tensor:
        assert self.prefix_len is not None
        assert attention_mask is None or 0 not in attention_mask
        assert not dropout

        if self.output_attentions:  # significant slowdown, use only for visualization or debugging
            self.attn_weights = calc_attn_weights(
                query_states=query_states,
                key_states=key_states,
                prefix_len=self.prefix_len,
                form=self.attention_weights_form,
            )
        else:
            self.attn_weights = None

        output = torch.concat((
            flash_attn_func(
                q=query_states[:, :self.prefix_len],
                k=key_states[:, :self.prefix_len],
                v=value_states[:, :self.prefix_len],
                causal=False),
            flash_attn_func(
                q=query_states[:, self.prefix_len:],
                k=key_states,
                v=value_states,
                causal=True),
        ), dim=1)

        if self.layer_idx + 1 == self.config.num_hidden_layers:
            PrefixUnmaskAttention.prefix_len = None

        return output


class PrefixUnmaskAdapter(AdapterBase):
    def get_args_kwargs(self,
                        input_ids: torch.Tensor,
                        _target_ids: torch.Tensor,
                        loss_mask: torch.Tensor,
                        _completion_mask: torch.Tensor,
                        _category_ids: torch.Tensor,
                        input_attn_mask: torch.Tensor,
                        target_attn_mask: torch.Tensor,
                        _metadata: BatchMetadata,
                        ) -> tuple[tuple[Any], dict[str, Any]]:
        if input_ids.shape[0] != 1:
            raise ValueError('This adapter only accepts batch_size = 1.')

        args = (input_ids,)
        kwargs = dict(attention_mask=input_attn_mask)
        PrefixUnmaskAttention.prefix_len = (~loss_mask & target_attn_mask).sum().item() + 1

        return args, kwargs

    def adapt(self, model: nn.Module) -> nn.Module:
        if model.config.attention_dropout != 0:
            raise ValueError('This adapter does not support the dropout.')
        elif model.device.type != 'cuda':
            raise ValueError('This adapter works with GPU only.')
        elif model.config._attn_implementation != AttentionImplementation.FA2:  # noqa: HF moment?
            raise ValueError('This adapter does not support attention implementations other than FA2.')

        for i, decoder_layer in enumerate(model.model.layers):
            decoder_layer.self_attn.__class__ = PrefixUnmaskAttention

            if decoder_layer.self_attn.layer_idx is None:
                decoder_layer.self_attn.layer_idx = i

        return model

    @staticmethod
    def set_attention_weights_form(model: nn.Module, form: AttentionWeightsForm) -> None:
        for decoder_layer in model.model.layers:
            decoder_layer.self_attn.attention_weights_form = form
