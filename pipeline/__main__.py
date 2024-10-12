from pipeline.data.composers.init import init_composer
from pipeline.data.dataset import train_test_split, set_transform
from pipeline.data.preprocessors.init import init_preprocessor
from pipeline.model.adapters.init import init_adapter
from pipeline.model.init import init_tokenizer_model
from pipeline.outputs.checkpointers.init import init_checkpointer
from pipeline.outputs.loggers.init import init_logger
from pipeline.outputs.metrics.init import init_metrics
from pipeline.trainers.init import init_trainer

import copy
import os
import sys

import hydra
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

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
    adapter_cls, checkpointer_cls, logger_cls, composer_cls, preprocessor_cls, trainer_cls = [
        os.path.dirname(config_choices.get(cfg_group))
        for cfg_group in ('adapter', 'checkpointer', 'logger', 'composer', 'preprocessor', 'trainer')
    ]

    checkpointer = init_checkpointer(
        cls_name=checkpointer_cls,
        loaded_config=config.checkpointer,
        directory=checkpoints_dir,
    )

    load_from = checkpointer.get_model_subdirectory()
    tokenizer, model = init_tokenizer_model(config.model, load_from=load_from)

    adapter = init_adapter(
        cls_name=adapter_cls,
        loaded_config=config.adapter,
        model_name=config.model.model_name)
    model = adapter.adapt(model)

    composer = init_composer(
        cls_name=composer_cls,
        loaded_config=config.composer,
        configs_dir=CONFIGS_DIR,
        tokenizer=tokenizer,
    )
    preprocessor = init_preprocessor(
        cls_name=preprocessor_cls,
        loaded_config=config.preprocessor,
        tokenizer=tokenizer,
    )

    if ('additional_composer' in config) != ('additional_preprocessor' in config):
        raise ValueError('Both or neither of additional_composer and '
                         'additional_preprocessor must be specified.')
    elif (
            (add_valid := ('additional_composer' in config)) and
            (config_choices.get('composer') == config.additional_composer) and
            (config_choices.get('preprocessor') == config.additional_preprocessor)
    ):
        raise ValueError('You are attempting to run validation twice '
                         'using the same data preprocessing steps.')

    add_composer = init_composer(
        cls_name=os.path.dirname(config.additional_composer),
        loaded_config=OmegaConf.load(
            os.path.join(CONFIGS_DIR, f'composer/{config.additional_composer}.yaml')
        ),
        configs_dir=CONFIGS_DIR,
        tokenizer=tokenizer,
    ) if add_valid else None

    add_preprocessor = init_preprocessor(
        cls_name=os.path.dirname(config.additional_preprocessor),
        loaded_config=OmegaConf.load(
            os.path.join(CONFIGS_DIR, f'preprocessor/{config.additional_preprocessor}.yaml')
        ),
        tokenizer=tokenizer,
    ) if add_valid else None

    logger = init_logger(
        cls_name=logger_cls,
        loaded_config=config.logger,
        directory=logs_dir,
        checkpointer=checkpointer,
        name=config.run_name,
        config=dict(config) | {'config_choices': config_choices, 'composer_initialization_code': repr(composer)},
    )
    if load_from is None:
        logger.message('The model is initialized from Hugging Face Hub.')
    else:
        logger.message(f'The model is initialized from {load_from}.')

    dataset = load_dataset(**dict(config.dataset))
    train_ds, valid_ds = train_test_split(dataset, **dict(config.split))
    add_valid_ds = copy.deepcopy(valid_ds) if add_valid else None

    set_transform(train_ds, composer, preprocessor)
    set_transform(valid_ds, composer, preprocessor)
    set_transform(add_valid_ds, add_composer, add_preprocessor)

    train_metrics = init_metrics(
        loaded_config=config.metrics.train_metrics,
        configs_dir=CONFIGS_DIR,
        tokenizer=tokenizer)
    valid_metrics = init_metrics(
        loaded_config=config.metrics.valid_metrics,
        configs_dir=CONFIGS_DIR,
        tokenizer=tokenizer,
    )

    trainer = init_trainer(
        cls_name=trainer_cls,
        loaded_config=config.trainer,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        add_valid_ds=add_valid_ds,
        adapter=adapter,
        checkpointer=checkpointer,
        logger=logger,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics)
    trainer.train(verbose=True)

    logger.message('Run successfully completed.')


if __name__ == '__main__':
    main()
