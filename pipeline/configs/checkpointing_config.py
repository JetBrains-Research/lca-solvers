from pipeline.configs.config_base import ConfigBase
from pipeline.outputs.checkpointing import LoadingMode
from pipeline.outputs.metrics.metrics_registry import MetricName

from dataclasses import dataclass
from typing import Callable


@dataclass
class CheckpointManagerConfig(ConfigBase):
    init_from: LoadingMode
    saving_freq: int
    main_metric: MetricName
    directory: str = 'checkpoints'

    # if you want to change it, override the following function accordingly
    checkpoint_directory_template: str = '{iteration_number:04d}'
    extract_iteration_number: Callable[[str], int] = staticmethod(int)

    model_subdirectory: str = 'model'
    optim_state_filename: str = 'optim.pt'
    metrics_filename: str = 'metrics.json'  # should be .json

    def __post_init__(self) -> None:
        self.init_from = LoadingMode(self.init_from)
