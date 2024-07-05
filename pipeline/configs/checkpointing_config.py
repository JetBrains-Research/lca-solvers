from __future__ import annotations

from pipeline.configs.config_base import ConfigBase
from pipeline.outputs.checkpointing import LoadingMode
from pipeline.outputs.metrics.metrics_registry import MetricName

from dataclasses import dataclass


@dataclass
class CheckpointManagerConfig(ConfigBase):
    init_from: LoadingMode
    saving_freq: int
    main_metric: MetricName
    directory: str = 'checkpoints'

    # if you want to change it, override the following function accordingly
    checkpoint_directory_template: str = '{iteration_number:04d}'
    extract_iteration_number = staticmethod(int)

    model_state_filename: str = 'model.pt'
    optim_state_filename: str = 'optim.pt'
    metrics_filename: str = 'metrics.json'  # should be .json

    @classmethod
    def from_yaml(cls, path: str) -> CheckpointManagerConfig:
        config = super().from_yaml(path)
        config.init_from = LoadingMode(config.init_from)
        return config  # noqa: PyCharm bug
