# config classes
from pipeline.configs.checkpointer_config import TopKCheckpointManagerConfig
from pipeline.configs.composer_config import ComposerConfig
from pipeline.configs.dataset_config import DatasetConfig
from pipeline.configs.logger_config import WandbLoggerConfig
from pipeline.configs.model_config import ModelConfig
from pipeline.configs.preprocessor_config import PreprocessorConfig
from pipeline.configs.split_config import SplitConfig
from pipeline.configs.trainer_config import FullFineTuningTrainerConfig

# main classes
from pipeline.data.composers.composers import PathDistanceComposer
from pipeline.data.preprocessors.file_level_preprocessor import FileLevelPreprocessor
from pipeline.outputs.checkpointers.top_k_checkpointer import TopKCheckpointManager
from pipeline.outputs.loggers.wandb_logger import WandbLogger
from pipeline.trainers.full_finetuning_trainer import FullFineTuningTrainer

# main functions
from pipeline.model.init import init_tokenizer_model
from pipeline.data.dataset import train_test_split, set_transform

from datasets import load_dataset


def main() -> None:
    # configs
    checkpointer_config = TopKCheckpointManagerConfig.from_yaml()
    composer_config = ComposerConfig.from_yaml()
    dataset_config = DatasetConfig.from_yaml()
    logger_config = WandbLoggerConfig.from_yaml()
    model_config = ModelConfig.from_yaml()
    preprocessor_config = PreprocessorConfig.from_yaml()
    split_config = SplitConfig.from_yaml()
    trainer_config = FullFineTuningTrainerConfig.from_yaml()

    # checkpointing
    checkpointer = TopKCheckpointManager(**checkpointer_config.dict)

    # logging
    logger_config.config = {
        'checkpointer': {'name': TopKCheckpointManager.__name__, **checkpointer_config.dict},
        'model': model_config.dict,
        'dataset': dataset_config.dict,
        'split': split_config.dict,
        'composer': {'name': PathDistanceComposer.__name__, **composer_config.dict},
        'preprocessor': {'name': FileLevelPreprocessor.__name__, **preprocessor_config.dict},
        'logger': {'name': WandbLogger.__name__, **logger_config.dict},
        'trainer': {'name': FullFineTuningTrainer.__name__, **trainer_config.dict},
    }
    logger = WandbLogger(checkpointer, **logger_config.dict)

    # tokenizer and model
    load_from = checkpointer.get_model_subdirectory()
    if load_from is None:
        logger.message('The model is initialized from Hugging Face Hub.')
    else:
        logger.message(f'The model is initialized from {load_from}.')
    tokenizer, model = init_tokenizer_model(load_from, **model_config.dict)

    # composer and preprocessor
    composer = PathDistanceComposer(**composer_config.dict)
    preprocessor = FileLevelPreprocessor(tokenizer, **preprocessor_config.dict)

    # dataset
    dataset = load_dataset(**dataset_config.dict)
    train_ds, valid_ds = train_test_split(dataset, **split_config.dict)
    set_transform(train_ds, valid_ds, composer, preprocessor)

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
