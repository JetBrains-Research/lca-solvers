# TODO: manage imports
from pipeline.configs.dataset_config import DatasetConfig

import math
import random
from collections import defaultdict
from typing import Literal

from datasets import load_dataset, Dataset


def load_dataset_with_config(config: DatasetConfig | Literal['train', 'small', 'medium', 'large', 'huge']) -> Dataset:
    if not isinstance(config, DatasetConfig):
        config = DatasetConfig.from_yaml(DatasetConfig.get_path2config(config))
    return load_dataset(**config.dict)


def train_test_split(dataset: Dataset,
                     test_size: float | int = 128,
                     upper_bound_per_repo: int = 5,  # TODO: variable name
                     random_seed: int | None = None,
                     ) -> tuple[Dataset, Dataset]:

    queue = defaultdict(list)
    repos_enum = list(enumerate(dataset['repo']))
    random.Random(random_seed).shuffle(repos_enum)

    for idx, repo in repos_enum:
        queue[repo].append(idx)

    queue = list(queue.items())
    random.Random(random_seed).shuffle(queue)
    # TODO: test correlation on number of samples per repo and its position in queue - must be 0

    if isinstance(test_size, float):
        test_size = math.ceil(test_size * len(queue))

    test_repos = set()
    test_repos_ids = list()
    cur_test_size = 0

    while cur_test_size != test_size:
        repo, ids = queue.pop()
        n_new_samples = min(upper_bound_per_repo, test_size - cur_test_size, len(ids))

        test_repos.add(repo)
        test_repos_ids.extend(ids[:n_new_samples])
        cur_test_size += n_new_samples

    train_ds = dataset.filter(lambda x: x['repo'] not in test_repos)
    test_ds = dataset.select(test_repos_ids)

    return train_ds, test_ds  # TODO: show disjointness


def stratified_train_test_split() -> tuple[Dataset, Dataset]:
    pass  # TODO: Can it provide less biased estimates? What strategies can be used?
