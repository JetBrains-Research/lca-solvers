from pipeline.configs.configs_registry import CONFIGS_REGISTRY
from pipeline.model.adapters.adapter_base import AdapterBase
from pipeline.model.adapters.adapters_registry import ADAPTERS_REGISTRY

from omegaconf import DictConfig


def init_adapter(cls_name: str, loaded_config: DictConfig, **kwargs) -> AdapterBase:
    config = CONFIGS_REGISTRY[cls_name].from_dict(dict(loaded_config) | kwargs)
    adapter = ADAPTERS_REGISTRY[cls_name](**config.dict)
    return adapter
