# TODO: manage imports
from pipeline.configs.raw_dataset_config import RawDatasetConfig

import math
import random

from datasets import load_dataset, Dataset
from huggingface_hub import HfFileSystem


def load_raw_test_dataset(ds_config: RawDatasetConfig) -> Dataset:
    return load_dataset(**ds_config.__dict__)


def load_raw_train_valid_datasets(ds_config: RawDatasetConfig, *,
                                  valid_size: float | int = 0,  # in chunks
                                  shuffle: bool = False,
                                  random_seed: int | None = None,
                                  ) -> tuple[Dataset, ...]:  # train + (valid)
    assert (((shuffle or random_seed is None) and valid_size != 0) or
            ((not shuffle and random_seed is None) and valid_size == 0))

    if valid_size == 0:
        return load_dataset(**ds_config.__dict__),

    fs = HfFileSystem()
    chunks = fs.ls(f'datasets/{ds_config.path}/train', detail=False)
    chunks = [chunk.split(fs.sep)[-1] for chunk in chunks]

    if isinstance(valid_size, float):
        valid_size = math.ceil(valid_size * len(chunks))  # >= 1

    if shuffle:
        random.Random(random_seed).shuffle(chunks)

    valid_chunks = chunks[:valid_size]
    train_chunks = chunks[valid_size:]

    return load_dataset(  # TODO: test disjointness
        **ds_config.__dict__,
        data_files=train_chunks,
    ), load_dataset(
        **ds_config.__dict__,
        data_files=valid_chunks,
    )


