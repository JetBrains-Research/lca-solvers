from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import LOGGER_YAML

from dataclasses import dataclass


@dataclass
class LocalLoggerConfig(ConfigBase):
    _default_path = LOGGER_YAML

    freq: int


@dataclass
class WandbLoggerConfig(ConfigBase):
    _default_path = LOGGER_YAML
