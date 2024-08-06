from pipeline.configs.configs_registry import CONFIGS_REGISTRY
from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.checkpointers.checkpointers_registry import CHECKPOINTERS_REGISTRY

from omegaconf import DictConfig


def init_checkpointer(cls_name: str, loaded_config: DictConfig, **kwargs) -> CheckpointManager:
    config = CONFIGS_REGISTRY[cls_name].from_dict(dict(loaded_config) | kwargs)
    checkpointer = CHECKPOINTERS_REGISTRY[cls_name](**config.dict)
    return checkpointer
