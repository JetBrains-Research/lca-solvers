from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import SPLIT_YAML

from dataclasses import dataclass


@dataclass
class SplitConfig(ConfigBase):
    _default_path = SPLIT_YAML

    test_size: int  # 0 means no validation at all
    upper_bound_per_repo: int
    random_seed: int | None
