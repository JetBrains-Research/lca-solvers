import sys
import importlib.util
import os
from types import ModuleType

from pipeline.configs.trainer_config import FullFineTuningTrainerConfig


# TODO: reorder all global constants
RUNS_DIR = 'runs'
TEMPLATE_DIR = os.path.join(RUNS_DIR, '.template')

# run directory content
CHECKPOINTS_DIR = 'checkpoints'
CONFIGS_DIR = 'configs'
LOGS_DIR = 'logs'
MODULE_NAME = 'run.py'


def import_module(run_name: str) -> ModuleType:
    path = os.path.join(RUNS_DIR, run_name, MODULE_NAME)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    run_name = 'test'
    run = import_module(run_name)
    run.main()


if __name__ == '__main__':
    main()
