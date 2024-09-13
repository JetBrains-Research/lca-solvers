from cache_study.utils import get_free_device, get_optimal_dtype

import copy
from enum import Enum

import torch
import torch.nn as nn
from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class AttentionImplementation(str, Enum):
    FA2 = 'flash_attention_2'
    SDPA = 'sdpa'
    EAGER = 'eager'


def get_optimal_attn(model_name: str, device: torch.device, dtype: torch.dtype) -> AttentionImplementation:
    hf_model_config = AutoConfig.from_pretrained(model_name)
    model_cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(hf_model_config)]

    fa2_supported = (
            is_flash_attn_2_available() and
            model_cls._supports_flash_attn_2 and  # noqa: HF doesn't have an API for this case
            device.type == 'cuda' and
            dtype in (torch.float16, torch.bfloat16)
    )

    if fa2_supported:
        return AttentionImplementation.FA2
    elif is_torch_sdpa_available() and model_cls._supports_sdpa:  # noqa: same
        return AttentionImplementation.SDPA
    else:
        return AttentionImplementation.EAGER


def init_tokenizer_model(model_name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    device = get_free_device()
    dtype = get_optimal_dtype()
    attn_implementation = get_optimal_attn(model_name, device, dtype)
    torch.set_float32_matmul_precision('high')

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True)
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    ).eval().requires_grad_(False)

    return tokenizer, model


def create_model_reference(model: nn.Module) -> nn.Module:
    model_ref = copy.deepcopy(model)
    for orig_param, ref_param in zip(model.parameters(), model_ref.parameters()):
        ref_param.data = orig_param.data
    return model_ref


def split_model(model: nn.Module, num_gen_layers: int) -> tuple[nn.Module, nn.Module]:
    encoder = create_model_reference(model.model)
    encoder.layers = encoder.layers[:-num_gen_layers]
    encoder.norm = nn.Identity()

    generator = create_model_reference(model)
    generator.model.layers = generator.model.layers[-num_gen_layers:]

    return encoder, generator
