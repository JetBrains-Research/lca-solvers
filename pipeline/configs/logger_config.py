from pipeline.configs.config_base import Config

from dataclasses import dataclass


@dataclass
class LoggerConfig(Config):
    freq: int


@dataclass
class WandbLoggerConfig(Config):
    pass
