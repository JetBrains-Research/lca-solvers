from pipeline.configs.configs_registry import CONFIGS_REGISTRY
from pipeline.trainers.trainer_base import TrainerBase
from pipeline.trainers.trainers_registry import TRAINERS_REGISTRY

from omegaconf import DictConfig


def init_trainer(cls_name: str, loaded_config: DictConfig, **kwargs) -> TrainerBase:
    config = CONFIGS_REGISTRY[cls_name].from_dict(dict(loaded_config) | kwargs)
    trainer = TRAINERS_REGISTRY[cls_name](**config.dict)
    return trainer
