from pipeline.model.adapters.adapter_base import AdapterBase

import copy
import re
from typing import Any, NoReturn
from typing_extensions import Self

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class ForbiddenModule(nn.Module):
    def forward(self, *_args, **_kwargs) -> NoReturn:
        raise RuntimeError('This module must not be called.')


def create_model_reference(model: nn.Module) -> nn.Module:
    model_ref = copy.deepcopy(model)
    for orig_param, ref_param in zip(model.parameters(), model_ref.parameters()):
        ref_param.data = orig_param.data
    return model_ref


def split_model(model: nn.Module, num_gen_layers: int | None) -> tuple[nn.Module | None, nn.Module]:
    if num_gen_layers is None:
        return None, model

    encoder = create_model_reference(model.model)
    encoder.layers = encoder.layers[:-num_gen_layers]
    encoder.norm = nn.Identity()

    generator = create_model_reference(model)
    generator.model.embed_tokens = ForbiddenModule()
    generator.model.layers = generator.model.layers[-num_gen_layers:]

    return encoder, generator


class CombinedModel(nn.Module):
    def __init__(self, encoder: nn.Module, generator: nn.Module, freeze_encoder: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.freeze_encoder = freeze_encoder

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.dtype

    def train(self, mode: bool = True) -> Self:
        if self.freeze_encoder:
            self.training = mode
            self.generator.train(mode)
        else:
            super().train(mode)

        return self

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        if self.freeze_encoder:
            self.generator.requires_grad_(requires_grad)
        else:
            super().requires_grad_(requires_grad)

        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> CausalLMOutputWithPast:
        with torch.inference_mode(self.freeze_encoder):
            encoder_output = self.encoder(input_ids, attention_mask)
            hidden_states = encoder_output.last_hidden_state

            if attention_mask is not None:
                inputs_embeds = hidden_states[attention_mask.bool()]
            else:
                inputs_embeds = hidden_states.flatten(0, 1)
            inputs_embeds = inputs_embeds.unsqueeze(0)  # restore batch dimension

        return self.generator(inputs_embeds=inputs_embeds)


class SplitAdapter(AdapterBase):
    def __init__(self, num_gen_layers: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_gen_layers = num_gen_layers

    def get_args_kwargs(self,
                        input_ids: torch.Tensor,
                        target_ids: torch.Tensor,
                        loss_mask: torch.Tensor,
                        completion_mask: torch.Tensor,
                        category_ids: torch.Tensor,
                        input_attn_mask: torch.Tensor,
                        target_attn_mask: torch.Tensor,
                        ) -> tuple[tuple[Any], dict[str, Any]]:
        # TODO
        args = (input_ids,)
        kwargs = dict(attention_mask=input_attn_mask)
        return args, kwargs

    def adapt(self, model: nn.Module) -> nn.Module:
        encoder, generator = split_model(model, self.num_gen_layers)

        freeze_encoder = not any(
            re.search(self.params_pattern, f'encoder.{name}')
            for name, _ in encoder.named_parameters()
        )
        if freeze_encoder:
            encoder = encoder.eval().requires_grad_(False)

        return CombinedModel(encoder, generator, freeze_encoder)
