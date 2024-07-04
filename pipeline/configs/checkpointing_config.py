from pipeline.configs.config_base import Config
from pipeline.outputs.checkpointing import LoadingMode

from dataclasses import dataclass


@dataclass
class ModelCheckpointConfig(Config):
    init_from: LoadingMode
    freq: int
    directory: str = 'checkpoints'
