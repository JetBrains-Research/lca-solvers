from pipeline.configs.checkpointer_config import (
    CheckpointManagerConfig,
    TopKCheckpointManagerConfig,
)
from pipeline.configs.composer_config import ChainedComposerConfig
from pipeline.configs.config_base import ConfigBase
from pipeline.configs.logger_config import (
    LocalLoggerConfig,
    WandbLoggerConfig,
)
from pipeline.configs.preprocessor_config import PreprocessorConfig
from pipeline.configs.trainer_config import FullFineTuningTrainerConfig

CONFIGS_REGISTRY = {
    # checkpointers
    'checkpointer': CheckpointManagerConfig,
    'top_k_checkpointer': TopKCheckpointManagerConfig,

    # loggers
    'dummy': ConfigBase,
    'local': LocalLoggerConfig,
    'wandb': WandbLoggerConfig,

    # composers
    'chained_composer': ChainedComposerConfig,

    # preprocessors
    'completion_loss_preprocessor': PreprocessorConfig,
    'file_level_preprocessor': PreprocessorConfig,
    'lm_preprocessor': PreprocessorConfig,

    # trainers
    'full_finetuning_trainer': FullFineTuningTrainerConfig,
}
