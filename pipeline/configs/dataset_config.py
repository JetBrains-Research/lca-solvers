from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import DATASET_YAML

from dataclasses import dataclass

from datasets import config


@dataclass
class DatasetConfig(ConfigBase):
    _default_path = DATASET_YAML

    path: str
    name: str | None = None
    data_dir: str | None = None
    split: str | None = None
    cache_dir: str = str(config.HF_DATASETS_CACHE)
