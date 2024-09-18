from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class AdapterConfig(ConfigBase):
    model_name: str
    params_pattern: str | None  # None means that all parameters receive updates
