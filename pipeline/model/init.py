from pipeline.configs.model_config import ModelConfig
from pipeline.environment.hardware import get_free_device, get_optimal_dtype

from enum import Enum

import torch
from omegaconf import DictConfig
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
    # nondeterministic
    FA2 = 'flash_attention_2'
    SDPA = 'sdpa'
    # deterministic
    EAGER = 'eager'


def init_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.tokenizer_name,
        trust_remote_code=config.trust_remote_code,
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


def init_model(config: ModelConfig) -> PreTrainedModel:
    if config.device is None:
        config.device = get_free_device()
    if config.dtype is None:
        config.dtype = get_optimal_dtype()
    if config.attn_implementation is None:
        config.attn_implementation = get_optimal_attn(config.model_name, config.device, config.dtype)
    if config.load_from is None:
        config.load_from = config.model_name

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.load_from,
        trust_remote_code=config.trust_remote_code,
        device_map=config.device,
        torch_dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        use_cache=config.use_cache,
    )

    if config.compile:
        model = torch.compile(model)
    return model


def init_tokenizer_model(loaded_config: DictConfig, **kwargs) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    config = ModelConfig.from_dict(dict(loaded_config) | kwargs)
    return init_tokenizer(config), init_model(config)
