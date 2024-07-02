from pipeline.configs.training_config import TrainingConfig

import abc


class TrainerBase(abc.ABC):
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
