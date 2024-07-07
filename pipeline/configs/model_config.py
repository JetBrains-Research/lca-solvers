from pipeline.configs.config_base import ConfigBase
from pipeline.model.init import AttentionImplementation

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig(ConfigBase):
    tokenizer_name: str
    model_name: str
    trust_remote_code: bool

    device: torch.device | None = None
    dtype: torch.dtype | None = None
    attn_implementation: AttentionImplementation | None = None
    compile: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

        if self.attn_implementation is not None:
            self.attn_implementation = AttentionImplementation(self.attn_implementation)
