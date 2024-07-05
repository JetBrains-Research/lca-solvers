from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class ModelConfig(ConfigBase):
    tokenizer_name: str
    model_name: str
    trust_remote_code: bool
    compile: bool
