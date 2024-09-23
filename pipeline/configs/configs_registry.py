from pipeline.configs.adapter_config import (
    AdapterConfig,
    SplitAdapterConfig,
)
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
    # adapters
    'identity_adapter': AdapterConfig,
    'prefix_unmask_adapter': AdapterConfig,
    'smooth_prefix_unmask_adapter': AdapterConfig,
    'split_adapter': SplitAdapterConfig,

    # checkpointers
    'checkpointer': CheckpointManagerConfig,
    'top_k_checkpointer': TopKCheckpointManagerConfig,

    # loggers
    'dummy_logger': ConfigBase,
    'local_logger': LocalLoggerConfig,
    'wandb_logger': WandbLoggerConfig,

    # composers
    'chained_composer': ChainedComposerConfig,

    # preprocessors
    'completion_loss_preprocessor': PreprocessorConfig,
    'file_level_preprocessor': PreprocessorConfig,
    'lm_preprocessor': PreprocessorConfig,

    # trainers
    'full_finetuning_trainer': FullFineTuningTrainerConfig,
}
