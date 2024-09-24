from pipeline.model.adapters.adapter_base import AdapterBase
from pipeline.model.init import AttentionImplementation

import re
from typing import Any
from typing_extensions import Self

import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import LlamaFlashAttention2


class SmoothPrefixUnmaskAttention(LlamaFlashAttention2):
    prefix_len = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.past_weight = None
        self.init_past_weight()

    def init_past_weight(self) -> None:
        self.past_weight = nn.Parameter(torch.ones(
            1, 1, self.config.num_attention_heads, 1,
            dtype=torch.float32,  # the precision of bfloat16 is not sufficient
            device=next(self.parameters()).device,
        ))

    def to(self, *args, **kwargs) -> Self:
        super(LlamaFlashAttention2, self).to(*args, **kwargs)
        self.past_weight.data = self.past_weight.data.to(torch.float32)
        return self

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

        output = flash_attn_func(
            q=query_states,
            k=key_states,
            v=value_states,
            causal=True)
        past_emb = output[:, :self.prefix_len]
        output = output.clone()

        future_emb = flash_attn_func(
            q=query_states[:, :self.prefix_len].flip(dims=(1,)),
            k=key_states[:, 1:self.prefix_len].flip(dims=(1,)),
            v=value_states[:, 1:self.prefix_len].flip(dims=(1,)),
            causal=True,
        ).flip(dims=(1,))

        past_weight = self.past_weight.to(output.dtype)
        # a bit more precise than "future_emb + past_weight * (past_emb - future_emb)"
        output[:, :self.prefix_len] = past_weight * past_emb + (1 - past_weight) * future_emb

        if self.layer_idx + 1 == self.config.num_hidden_layers:
            SmoothPrefixUnmaskAttention.prefix_len = None

        return output


class SmoothPrefixUnmaskAdapter(AdapterBase):
    def __init__(self, past_weight_decay: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.past_weight_decay = past_weight_decay

    def init_optimizer(self, model: nn.Module, **optim_kwargs) -> torch.optim.AdamW:
        weight_decay = optim_kwargs.pop('weight_decay', 0)

        past_weights = list()
        decay_params = list()
        no_decay_params = list()

        for name, params in model.named_parameters():
            if self.params_pattern is not None and not re.search(self.params_pattern, name):
                continue  # not trainable

            if name.endswith('past_weight'):
                past_weights.append(params)
            elif params.dim() >= 2:
                decay_params.append(params)
            else:
                no_decay_params.append(params)

        optimizer = torch.optim.AdamW(params=[
            {'name': 'past_weights', 'params': past_weights, 'weight_decay': self.past_weight_decay},
            {'name': 'decay_params', 'params': decay_params, 'weight_decay': weight_decay},
            {'name': 'no_decay_params', 'params': no_decay_params, 'weight_decay': 0},
        ], **optim_kwargs)

        @torch.inference_mode
        def proj_onto_budget_set(*_args, **_kwargs) -> None:
            for past_weight in past_weights:
                past_weight.clamp_(0, 1)

        optimizer.register_step_post_hook(proj_onto_budget_set)
        return optimizer

    def get_args_kwargs(self,
                        input_ids: torch.Tensor,
                        _target_ids: torch.Tensor,
                        loss_mask: torch.Tensor,
                        _completion_mask: torch.Tensor,
                        _category_ids: torch.Tensor,
                        input_attn_mask: torch.Tensor,
                        target_attn_mask: torch.Tensor,
                        ) -> tuple[tuple[Any], dict[str, Any]]:
        if input_ids.shape[0] != 1:
            raise ValueError('This adapter only accepts batch_size = 1.')

        args = (input_ids,)
        kwargs = dict(attention_mask=input_attn_mask)
        SmoothPrefixUnmaskAttention.prefix_len = (~loss_mask & target_attn_mask).sum().item() + 1

        return args, kwargs

    def adapt(self, model: nn.Module) -> nn.Module:
        if model.config.attention_dropout != 0:
            raise ValueError('This adapter does not support the dropout.')
        elif model.device.type != 'cuda':
            raise ValueError('This adapter works with GPU only.')
        elif model.config._attn_implementation != AttentionImplementation.FA2:  # noqa: HF moment?
            raise ValueError('This adapter does not support attention implementations other than FA2.')

        for i, decoder_layer in enumerate(model.model.layers):
            decoder_layer.self_attn.__class__ = SmoothPrefixUnmaskAttention
            decoder_layer.self_attn.init_past_weight()

            if decoder_layer.self_attn.layer_idx is None:
                decoder_layer.self_attn.layer_idx = i

        return model
