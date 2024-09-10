from pipeline.model.adapters.adapter_base import AdapterBase
from pipeline.model.init import AttentionImplementation

from typing import Any

import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import LlamaFlashAttention2


class PrefixUnmaskAttention(LlamaFlashAttention2):
    # see notebooks/attention_benchmarking.ipynb for more details
    prefix_len = None

    def __init__(self, is_last: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_last = is_last

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
                causal=True)
        ), dim=1)

        if self.is_last:
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
                        ) -> tuple[tuple[Any], dict[str, Any]]:
        if input_ids.shape[0] != 1:
            raise ValueError('This adapter only accepts batch_size = 1.')

        # TODO: slice test (just check correctness of the prefix_len)
        # TODO: bug - targets are not truncated at all
        # TODO: visualize attention scores [3 masking setups] (raw vs. trained) = table 2x3
        args = (input_ids[input_attn_mask.bool()].unsqueeze(0),)
        kwargs = dict(attention_mask=torch.ones_like(args[0]))
        PrefixUnmaskAttention.prefix_len = (~loss_mask & target_attn_mask).sum().item()

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
            decoder_layer.self_attn.is_last = (i + 1 == model.config.num_hidden_layers)

        return model
