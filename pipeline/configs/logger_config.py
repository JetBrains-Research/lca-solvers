from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class LocalLoggerConfig(ConfigBase):
    freq: int


@dataclass
class WandbLoggerConfig(ConfigBase):
    pass
