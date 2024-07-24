from pipeline.configs.config_base import ConfigBase
from pipeline.environment.run_directory import CHECKPOINTS_DIR, CHECKPOINTING_YAML
from pipeline.outputs.checkpointing import LoadingMode
from pipeline.outputs.metrics.metric_base import MetricName

from dataclasses import dataclass
from typing import Callable


@dataclass
class CheckpointManagerConfig(ConfigBase):
    _default_path = CHECKPOINTING_YAML

    init_from: LoadingMode | str
    main_metric: MetricName
    directory: str = CHECKPOINTS_DIR

    # if you want to change it, override the following function accordingly
    checkpoint_directory_template: str = '{iteration_number:04d}'
    extract_iteration_number: Callable[[str], int] = staticmethod(int)

    model_subdirectory: str = 'model'
    optim_state_filename: str = 'optim.pt'
    metrics_filename: str = 'metrics.json'  # should be .json

    def __post_init__(self) -> None:
        if self.init_from in set(LoadingMode):
            self.init_from = LoadingMode(self.init_from)


@dataclass(kw_only=True)
class TopKCheckpointManagerConfig(CheckpointManagerConfig):
    max_checkpoints_num: int
