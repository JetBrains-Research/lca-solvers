from pipeline.configs.config_base import Config

from dataclasses import dataclass

from datasets import config


@dataclass
class DatasetConfig(Config):
    path: str
    name: str | None = None
    data_dir: str | None = None
    split: str | None = None
    cache_dir: str = str(config.HF_DATASETS_CACHE)
