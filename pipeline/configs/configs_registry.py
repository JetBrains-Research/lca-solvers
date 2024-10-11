from pipeline.configs.adapter_config import (
    AdapterConfig,
    SmoothPrefixUnmaskAdapterConfig,
    SplitAdapterConfig,
)
from pipeline.configs.checkpointer_config import (
    CheckpointManagerConfig,
    TopKCheckpointManagerConfig,
)
from pipeline.configs.composer_config import (
    ChainedComposerConfig,
    SplitComposerConfig,
)
from pipeline.configs.config_base import ConfigBase
from pipeline.configs.logger_config import (
    LocalLoggerConfig,
    WandbLoggerConfig,
)
from pipeline.configs.preprocessor_config import (
    PreprocessorConfig,
    SplitPreprocessorConfig,
)
from pipeline.configs.trainer_config import UniversalTrainerConfig

CONFIGS_REGISTRY = {
    # adapters
    'identity_adapter': AdapterConfig,
    'prefix_unmask_adapter': AdapterConfig,
    'smooth_prefix_unmask_adapter': SmoothPrefixUnmaskAdapterConfig,
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
    'split_composer': SplitComposerConfig,

    # preprocessors
    'completion_loss_preprocessor': PreprocessorConfig,
    'file_level_preprocessor': PreprocessorConfig,
    'lm_preprocessor': PreprocessorConfig,
    'split_completion_loss_preprocessor': SplitPreprocessorConfig,
    'split_lm_preprocessor': SplitPreprocessorConfig,

    # trainers
    'universal_trainer': UniversalTrainerConfig,
}
