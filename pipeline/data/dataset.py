import math
import random
from collections import defaultdict

from datasets import Dataset


def train_test_split(dataset: Dataset,
                     test_size: int,  # TODO: edge case - 0
                     upper_bound_per_repo: int,
                     random_seed: int | None,
                     ) -> tuple[Dataset, Dataset | None]:
    generator = random.Random(random_seed)
    queue = defaultdict(list)
    repos_enum = list(enumerate(dataset['repo']))
    generator.shuffle(repos_enum)

    for idx, repo in repos_enum:
        queue[repo].append(idx)

    queue = list(queue.items())
    generator.shuffle(queue)
    # TODO: test correlation on number of samples per repo and its position in queue - must be 0

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

    return train_ds, test_ds  # TODO: show disjointness


def stratified_train_test_split() -> tuple[Dataset, Dataset]:
    pass  # TODO: Can it provide less biased estimates? What strategies can be used?
