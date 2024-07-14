from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import LOGS_DIR, LOGGER_YAML, get_run_name

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class LocalLoggerConfig(ConfigBase):
    _default_path = LOGGER_YAML

    train_csv: str
    valid_csv: str
    stdout_file: str
    stderr_file: str
    directory: str = LOGS_DIR


@dataclass(kw_only=True)
class WandbLoggerConfig(LocalLoggerConfig):
    _default_path = LOGGER_YAML

    project: str
    name: str = get_run_name()
    config: dict[str, Any] | None = None
    group: str | None = None
    notes: str | None = None
    resume: Literal['allow', 'must', 'never', 'auto']
