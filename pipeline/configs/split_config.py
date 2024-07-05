from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class SplitConfig(ConfigBase):
    test_size: int | float  # 0 means no validation at all
    upper_bound_per_repo: int
    random_seed: int | None
