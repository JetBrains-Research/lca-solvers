from pipeline.environment.hardware import get_free_device, get_optimal_dtype

from enum import Enum

import torch
import torch.nn as nn
from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerBase


class AttentionImplementation(str, Enum):
    FA2 = 'flash_attention_2'
    SDPA = 'sdpa'
    EAGER = 'eager'


def init_tokenizer(tokenizer_name: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name,
        trust_remote_code=trust_remote_code,
    )


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


def init_model(load_from: str | None,
               model_name: str,
               trust_remote_code: bool,
               use_cache: bool,
               device: torch.device,
               dtype: torch.dtype,
               attn_implementation: AttentionImplementation | None,
               compile: bool,  # noqa: built-in function that won't be used
               ) -> nn.Module:
    if device is None:
        device = get_free_device()
    if dtype is None:
        dtype = get_optimal_dtype()
    if attn_implementation is None:
        attn_implementation = get_optimal_attn(model_name, device, dtype)
    if load_from is None:
        load_from = model_name

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=load_from,
        trust_remote_code=trust_remote_code,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        use_cache=use_cache,
    )

    if compile:
        model = torch.compile(model)
    return model


def init_tokenizer_model(load_from: str | None,
                         tokenizer_name: str,
                         model_name: str,
                         trust_remote_code: bool,
                         use_cache: bool,
                         device: torch.device,
                         dtype: torch.dtype,
                         attn_implementation: AttentionImplementation | None,
                         compile: bool,  # noqa: built-in function that won't be used
                         ) -> tuple[PreTrainedTokenizerBase, nn.Module]:
    return (
        init_tokenizer(tokenizer_name, trust_remote_code),
        init_model(load_from, model_name, trust_remote_code, use_cache, device, dtype, attn_implementation, compile),
    )
