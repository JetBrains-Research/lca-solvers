import os
from functools import partial

RUNS_DIR = 'runs'
RUNS_SRC_DIR = os.path.join(RUNS_DIR, '.templates')
SRC_RUN_PY = 'standard.py'

# run directory content
CHECKPOINTS_DIR = 'checkpoints'
CONFIGS_DIR = 'configs'
LOGS_DIR = 'logs'
RUN_PY = 'run.py'
join_path = partial(os.path.join, CONFIGS_DIR)

# default destination configs
CHECKPOINTER_YAML = join_path('checkpointer.yaml')
COMPOSER_YAML = join_path('composer.yaml')
DATASET_YAML = join_path('dataset.yaml')
LOGGER_YAML = join_path('logger.yaml')
MODEL_YAML = join_path('model.yaml')
PREPROCESSOR_YAML = join_path('preprocessor.yaml')
SPLIT_YAML = join_path('split.yaml')
TRAINER_YAML = join_path('trainer.yaml')

# config templates directory
CONFIGS_SRC_DIR = 'configs'
join_path = partial(os.path.join, CONFIGS_SRC_DIR)

# default source directories
CHECKPOINTERS_SRC_DIR = join_path('checkpointers')
COMPOSERS_SRC_DIR = join_path('composers')
DATASETS_SRC_DIR = join_path('datasets')
LOGGERS_SRC_DIR = join_path('loggers')
MODELS_SRC_DIR = join_path('models')
PREPROCESSORS_SRC_DIR = join_path('preprocessors')
SPLITS_SRC_DIR = join_path('splits')
TRAINERS_SRC_DIR = join_path('trainers')

# default source configs
SRC_CHECKPOINTER_YAML = 'standard.yaml'
SRC_COMPOSER_YAML = 'empty.yaml'
SRC_DATASET_YAML = 'train.yaml'
SRC_LOGGER_YAML = 'wandb.yaml'
SRC_MODEL_YAML = 'dseek1p3.yaml'
SRC_PREPROCESSOR_YAML = 'lm_standard.yaml'
SRC_SPLIT_YAML = 'train_valid_split.yaml'
SRC_TRAINER_YAML = 'full_fine_tuning_standard.yaml'


def get_run_name() -> str:
    return os.path.basename(os.getcwd())
