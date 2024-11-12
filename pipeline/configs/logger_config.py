from pipeline.configs.config_base import ConfigBase
from pipeline.outputs.checkpointers.checkpointer import CheckpointManager

from dataclasses import dataclass
from typing import Any


@dataclass
class LocalLoggerConfig(ConfigBase):
    train_csv: str
    valid_csv: str
    stdout_file: str
    stderr_file: str
    directory: str


@dataclass
class WandbLoggerConfig(LocalLoggerConfig):
    checkpointer: CheckpointManager
    project: str
    name: str
    config: dict[str, Any] | None = None
    group: str | None = None
    notes: str | None = None
