from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass

from datasets import config


@dataclass
class DatasetConfig(ConfigBase):
    path: str
    name: str | None = None
    data_dir: str | None = None
    split: str | None = None
    cache_dir: str = str(config.HF_DATASETS_CACHE)
