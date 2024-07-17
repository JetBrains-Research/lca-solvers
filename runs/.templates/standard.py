# config classes
from pipeline.configs.checkpointing_config import CheckpointManagerConfig
from pipeline.configs.composer_config import ComposerConfig
from pipeline.configs.dataset_config import DatasetConfig
from pipeline.configs.logger_config import WandbLoggerConfig
from pipeline.configs.model_config import ModelConfig
from pipeline.configs.preprocessor_config import LMPreprocessorConfig
from pipeline.configs.split_config import SplitConfig
from pipeline.configs.trainer_config import FullFineTuningTrainerConfig

# main classes
from pipeline.data.composers.path_distance_composer import PathDistanceComposer
from pipeline.data.preprocessing.lm_preprocessor import LMPreprocessor
from pipeline.outputs.checkpointing import CheckpointManager
from pipeline.outputs.loggers.wandb_logger import WandbLogger
from pipeline.trainers.full_finetuning_trainer import FullFineTuningTrainer

# main functions
from pipeline.model.init import init_tokenizer_model
from pipeline.data.dataset import train_test_split

from datasets import load_dataset


def main() -> None:
    # checkpointing
    checkpointing_config = CheckpointManagerConfig.from_yaml()
    checkpointer = CheckpointManager(**checkpointing_config.dict)

    # tokenizer and model
    model_config = ModelConfig.from_yaml()
    load_from = checkpointer.get_model_subdirectory()
    tokenizer, model = init_tokenizer_model(load_from, **model_config.dict)

    # dataset and its split
    dataset_config = DatasetConfig.from_yaml()
    split_config = SplitConfig.from_yaml()
    dataset = load_dataset(**dataset_config.dict)
    train_ds, valid_ds = train_test_split(dataset, **split_config.dict)

    # composer
    composer_config = ComposerConfig.from_yaml()
    composer = PathDistanceComposer(**composer_config.dict)

    # preprocessor
    preprocessor_config = LMPreprocessorConfig.from_yaml()
    preprocessor = LMPreprocessor(tokenizer, **preprocessor_config.dict)

    # set dataset transform
    transform = lambda x: preprocessor(composer.compose_batch(x))
    train_ds.set_transform(transform)
    valid_ds.set_transform(transform)

    # trainer config
    trainer_config = FullFineTuningTrainerConfig.from_yaml()

    # logging
    logger_config = WandbLoggerConfig.from_yaml()
    logger_config.config = {
        'checkpointing': checkpointing_config.dict,
        'model': model_config.dict,
        'dataset': dataset_config.dict,
        'split': split_config.dict,
        'composer': {'name': PathDistanceComposer.__name__, **composer_config.dict},
        'preprocessor': preprocessor_config.dict,
        'logger': logger_config.dict,
        'trainer': trainer_config.dict,
    }
    logger = WandbLogger(**logger_config.dict)

    trainer = FullFineTuningTrainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        checkpointer=checkpointer,
        logger=logger,
        **trainer_config.dict)
    trainer.train(verbose=True)

    logger.message('Run successfully completed.')


if __name__ == '__main__':
    main()
