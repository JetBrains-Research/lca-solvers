from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal

import yaml
from datasets import config


@dataclass
class DatasetConfig:
    path: str
    name: str | None = None
    data_dir: str | None = None
    split: str | None = None
    cache_dir: str = str(config.HF_DATASETS_CACHE)

    dict = property(asdict)

    @staticmethod
    def get_path2config(name: Literal['train', 'small', 'medium', 'large', 'huge']) -> str:
        return f'configs/datasets/{name}.yaml'

    @staticmethod
    def from_yaml(path: str | None) -> DatasetConfig:
        if path is None:
            path = DatasetConfig.get_path2config('train')

        with open(path) as stream:
            return DatasetConfig(**yaml.safe_load(stream))
