from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import LOGS_DIR, LOGGER_YAML

from dataclasses import dataclass


@dataclass
class LocalLoggerConfig(ConfigBase):
    _default_path = LOGGER_YAML

    train_csv: str
    valid_csv: str
    stdout_file: str
    stderr_file: str
    directory: str = LOGS_DIR


@dataclass
class WandbLoggerConfig(LocalLoggerConfig):
    _default_path = LOGGER_YAML

    # TODO
