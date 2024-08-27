from pipeline.data.composers.init import init_composer
from pipeline.data.dataset import train_test_split, set_transform
from pipeline.data.preprocessors.init import init_preprocessor
from pipeline.model.init import init_tokenizer_model
from pipeline.outputs.checkpointers.init import init_checkpointer
from pipeline.outputs.loggers.init import init_logger
from pipeline.trainers.init import init_trainer

import os
import sys

import hydra
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

LCA_SOLVERS_DIR = os.path.dirname(os.path.dirname(__file__))

# configs
CONFIGS_DIR = os.path.join(LCA_SOLVERS_DIR, 'configs')
MAIN_CONFIG = 'defaults'

# run directory
RUNS_DIR = os.path.join(LCA_SOLVERS_DIR, 'runs')
ARGV_SH_FILE = 'run_script.sh'
CHECKPOINTS_DIR = 'checkpoints'
LOGS_DIR = 'logs'


# TODO: unify init functions


@hydra.main(config_path=CONFIGS_DIR, config_name=MAIN_CONFIG, version_base=None)
def main(config: DictConfig) -> None:
    argv_sh = ' \\\n'.join([sys.executable] + sys.argv)

    run_dir = os.path.join(RUNS_DIR, config.run_name)
    argv_sh_file = os.path.join(run_dir, ARGV_SH_FILE)
    checkpoints_dir = os.path.join(run_dir, CHECKPOINTS_DIR)
    logs_dir = os.path.join(run_dir, LOGS_DIR)

    if os.path.exists(run_dir):
        with open(argv_sh_file) as stream:
            old_argv_sh = stream.read()

        if argv_sh != old_argv_sh:
            input(f'Mismatch of script arguments with an older instance of the same run ({argv_sh_file}).\n'
                  'Press ENTER to continue with the new ones.')
    else:
        os.mkdir(run_dir)
        os.mkdir(checkpoints_dir)
        os.mkdir(logs_dir)

        os.mknod(os.path.join(checkpoints_dir, '.gitkeep'))
        os.mknod(os.path.join(logs_dir, '.gitkeep'))

    with open(argv_sh_file, 'w') as stream:
        stream.write(argv_sh)

    config_choices = HydraConfig.get().runtime.choices
    checkpointer_cls, logger_cls, composer_cls, preprocessor_cls, trainer_cls = [
        os.path.dirname(config_choices.get(cfg_group))
        for cfg_group in ('checkpointer', 'logger', 'composer', 'preprocessor', 'trainer')
    ]

    checkpointer = init_checkpointer(
        cls_name=checkpointer_cls,
        loaded_config=config.checkpointer,
        directory=checkpoints_dir,
    )

    composer = init_composer(
        cls_name=composer_cls,
        loaded_config=config.composer,
        configs_dir=CONFIGS_DIR,
    )

    logger = init_logger(
        cls_name=logger_cls,
        loaded_config=config.logger,
        directory=logs_dir,
        checkpointer=checkpointer,
        name=config.run_name,
        config=dict(config) | {'config_choices': config_choices, 'composer_initialization_code': repr(composer)},
    )

    load_from = checkpointer.get_model_subdirectory()
    if load_from is None:
        logger.message('The model is initialized from Hugging Face Hub.')
    else:
        logger.message(f'The model is initialized from {load_from}.')
    tokenizer, model = init_tokenizer_model(config.model, load_from=load_from)

    preprocessor = init_preprocessor(
        cls_name=preprocessor_cls,
        loaded_config=config.preprocessor,
        tokenizer=tokenizer,
    )

    dataset = load_dataset(**dict(config.dataset))
    train_ds, valid_ds = train_test_split(dataset, **dict(config.split))
    set_transform(train_ds, valid_ds, composer, preprocessor)

    trainer = init_trainer(
        cls_name=trainer_cls,
        loaded_config=config.trainer,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        checkpointer=checkpointer,
        logger=logger)
    trainer.train(verbose=True)

    logger.message('Run successfully completed.')


if __name__ == '__main__':
    main()
