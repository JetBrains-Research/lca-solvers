from dataclasses import dataclass

from lca_eval_harness.data_classes.datapoint_base import DataPointBase


@dataclass
class DataPointPrefixIndexes(DataPointBase):
    prefix_split_indexes: list[int]
    # prefix_line_split_indexes: list[int]

    def split_completion_file(self) -> (list[str], list[str]):
        prefixes = list()
        postfixes = list()
        for split_idx in self.prefix_split_indexes:
            if split_idx < 0:
                prefixes.append('')
                postfixes.append(self.completion_file)
            elif split_idx > len(self.completion_file):
                prefixes.append(self.completion_file)
                postfixes.append('')
            else:
                prefixes.append(self.completion_file[:split_idx])
                postfixes.append(self.completion_file[split_idx:])
        return prefixes, postfixes



