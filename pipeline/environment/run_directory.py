import os

# TODO: reorder all global constants
RUNS_DIR = 'runs'
TEMPLATE_DIR = os.path.join(RUNS_DIR, '.template')

# run directory content
CHECKPOINTS_DIR = 'checkpoints'
CONFIGS_DIR = 'configs'
LOGS_DIR = 'logs'
RUN_FILE = 'run.py'

# default config files
CHECKPOINTING_YAML = os.path.join(CONFIGS_DIR, 'checkpointing.yaml')
COMPOSER_YAML = os.path.join(CONFIGS_DIR, 'composer.yaml')
DATASET_YAML = os.path.join(CONFIGS_DIR, 'dataset.yaml')
LOGGER_YAML = os.path.join(CONFIGS_DIR, 'logger.yaml')
MODEL_YAML = os.path.join(CONFIGS_DIR, 'model.yaml')
PREPROCESSOR_YAML = os.path.join(CONFIGS_DIR, 'preprocessor.yaml')
SPLIT_YAML = os.path.join(CONFIGS_DIR, 'split.yaml')
TRAINER_YAML = os.path.join(CONFIGS_DIR, 'trainer.yaml')


def get_run_name() -> str:
    return os.path.basename(os.getcwd())
