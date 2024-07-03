from pipeline.configs.config_base import Config

from dataclasses import dataclass


@dataclass
class ModelConfig(Config):
    tokenizer_name: str
    model_name: str
