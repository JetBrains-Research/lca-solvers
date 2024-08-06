from pipeline.configs.configs_registry import CONFIGS_REGISTRY
from pipeline.data.preprocessors.preprocessor_base import PreprocessorBase
from pipeline.data.preprocessors.preprocessors_registry import PREPROCESSORS_REGISTRY

from omegaconf import DictConfig


def init_preprocessor(cls_name: str, loaded_config: DictConfig, **kwargs) -> PreprocessorBase:
    config = CONFIGS_REGISTRY[cls_name].from_dict(dict(loaded_config) | kwargs)
    preprocessor = PREPROCESSORS_REGISTRY[cls_name](**config.dict)
    return preprocessor
