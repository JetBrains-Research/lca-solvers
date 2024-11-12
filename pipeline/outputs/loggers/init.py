from pipeline.configs.configs_registry import CONFIGS_REGISTRY
from pipeline.outputs.loggers.logger_base import LoggerBase
from pipeline.outputs.loggers.loggers_registry import LOGGERS_REGISTRY

from omegaconf import DictConfig


def init_logger(cls_name: str, loaded_config: DictConfig, **kwargs) -> LoggerBase:
    config = CONFIGS_REGISTRY[cls_name].from_dict(dict(loaded_config) | kwargs)
    logger = LOGGERS_REGISTRY[cls_name](**config.dict)
    return logger
