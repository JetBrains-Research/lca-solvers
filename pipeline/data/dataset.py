from pipeline.data.composers.base_composers import ComposerBase
from pipeline.data.preprocessing.preprocessor_base import PreprocessorBase

import random
from collections import defaultdict

from datasets import Dataset


def train_test_split(dataset: Dataset,
                     test_size: int,
                     upper_bound_per_repo: int,
                     random_seed: int | None = None,
                     ) -> tuple[Dataset, Dataset | None]:
    if test_size == 0:
        return dataset, None

    generator = random.Random(random_seed)
    queue = defaultdict(list)
    repos_enum = list(enumerate(dataset['repo']))
    generator.shuffle(repos_enum)

    for idx, repo in repos_enum:
        queue[repo].append(idx)

    queue = list(queue.items())
    generator.shuffle(queue)

    train_repos_ids = set(range(len(dataset)))
    test_repos_ids = set()
    cur_test_size = 0

    while cur_test_size != test_size:
        if queue:
            repo, ids = queue.pop()
        else:
            raise ValueError(
                'There are not enough data points in the original dataset to satisfy both the '
                'test_size and upper_bound_per_repo arguments. Try either decreasing the test_size '
                'or increasing the upper_bound_per_repo.')

        num_new_samples = min(upper_bound_per_repo, test_size - cur_test_size, len(ids))

        train_repos_ids.difference_update(ids)
        test_repos_ids.update(ids[:num_new_samples])
        cur_test_size += num_new_samples

    train_ds = dataset.select(train_repos_ids)
    test_ds = dataset.select(test_repos_ids)

    return train_ds, test_ds


def set_transform(train_ds: Dataset,
                  test_ds: Dataset | None,
                  composer: ComposerBase,
                  preprocessor: PreprocessorBase,
                  ) -> None:
    transform = lambda x: preprocessor(composer.compose_batch(x))
    train_ds.set_transform(transform)
    if test_ds is not None:
        test_ds.set_transform(transform)
