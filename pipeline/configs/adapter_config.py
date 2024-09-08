from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class AdapterConfig(ConfigBase):
    model_name: str
