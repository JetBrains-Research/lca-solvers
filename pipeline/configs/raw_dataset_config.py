from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import yaml


@dataclass
class RawDatasetConfig:
    path: str
    name: str | None
    data_dir: str | None
    split: str
    cache_dir: str
    streaming: bool

    @staticmethod
    def from_yaml(path: str = 'configs/datasets.yaml', *,
                  ds_name: str = 'train',
                  location: Literal['local', 'server'] = 'server',
                  ) -> RawDatasetConfig:

        with open(path) as stream:
            config_dict = yaml.safe_load(stream)

        return RawDatasetConfig(
            **config_dict['raw_datasets'][ds_name],
            cache_dir=config_dict['cache_dir'][location],
            streaming=(ds_name == 'train' and location == 'local'),
        )
