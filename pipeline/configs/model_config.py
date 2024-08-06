from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class ModelConfig(ConfigBase):
    tokenizer_name: str
    model_name: str
    trust_remote_code: bool
    load_from: str | None

    use_cache: bool = False
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    attn_implementation: Literal['flash_attention_2', 'sdpa', 'eager'] | None = None
    compile: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)
