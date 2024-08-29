from pipeline.configs.configs_registry import CONFIGS_REGISTRY
from pipeline.data.composers.blocks.blocks_registry import BLOCKS_REGISTRY
from pipeline.data.composers.composer_base import ComposerBase
from pipeline.data.composers.composers_registry import COMPOSERS_REGISTRY

import os

import yaml
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase


def init_composer(cls_name: str,
                  loaded_config: DictConfig,
                  configs_dir: str,
                  tokenizer: PreTrainedTokenizerBase,
                  **kwargs,
                  ) -> ComposerBase:
    config = CONFIGS_REGISTRY[cls_name].from_dict(dict(loaded_config) | kwargs)

    if cls_name == 'chained_composer':
        for path in loaded_config.block_configs:
            full_path = os.path.join(configs_dir, 'composer/chained_composer/blocks', path)
            block_name = os.path.basename(os.path.dirname(path))

            with open(full_path) as stream:
                block_config = yaml.safe_load(stream)

            if block_config is None:
                block_config = dict()

            block_cls = BLOCKS_REGISTRY[block_name]
            if block_cls.requires_tokenizer:
                block_config['tokenizer'] = tokenizer

            block = block_cls(**block_config)
            config.blocks.append(block)

    composer = COMPOSERS_REGISTRY[cls_name](**config.dict)
    return composer
