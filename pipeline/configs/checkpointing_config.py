from pipeline.configs.config_base import Config

from dataclasses import dataclass


@dataclass
class ModelCheckpointConfig(Config):
    freq: int
    directory: str
    init_from: str | None = None


@dataclass
class TopKModelCheckpointConfig(Config):
    pass
