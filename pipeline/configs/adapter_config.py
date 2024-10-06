from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class AdapterConfig(ConfigBase):
    model_name: str
    params_pattern: str | None  # None means that all parameters receive updates


@dataclass
class SmoothPrefixUnmaskAdapterConfig(AdapterConfig):
    past_weight_decay: float


@dataclass
class SplitAdapterConfig(AdapterConfig):
    num_gen_layers: int
    simplified_rope: bool
