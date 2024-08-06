from pipeline.data.composers.old.chunking_mixins import Chunk
from pipeline.data.datapoint import Datapoint

import os
import warnings
from typing import Sequence


class RankerMixin:
    @staticmethod
    def rank(chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[int | float]:
        return range(len(chunks))


class NegativePathDistance(RankerMixin):
    @staticmethod
    def _path_distance(path_from: str, path_to: str) -> int:
        path_from = os.path.normpath(path_from)
        path_to = os.path.normpath(path_to)

        if path_from == path_to:
            warnings.warn(f'Data leakage: the {path_from} completion file is contained in the repo snapshot.')

        divided_path_from = path_from.split(os.path.sep)
        divided_path_to = path_to.split(os.path.sep)

        common_len = 0
        for segment_from, segment_to in zip(divided_path_from, divided_path_to):
            if segment_from == segment_to:
                common_len += 1
            else:
                break

        num_residuals_from = len(divided_path_from) - common_len - 1
        num_residuals_to = len(divided_path_to) - common_len - 1

        return num_residuals_from + num_residuals_to

    def rank(self, chunks: Sequence[Chunk], datapoint: Datapoint) -> Sequence[int | float]:
        return [
            -self._path_distance(chunk.metadata['filename'], datapoint.completion_file['filename'])
            for chunk in chunks
        ]


class InverseGroupingPathDistance(NegativePathDistance):
    def __init__(self, ordered_groups: list[list[str]]) -> None:
        self.group_weights = {
            extension: weight
            for weight, group in enumerate(ordered_groups)
            for extension in group
        }

    def rank(self, chunks: Sequence[Chunk], datapoint: Datapoint) -> Sequence[int | float]:
        return [
            self.group_weights['.' + chunk.metadata['filename'].split('.')[-1]] +
            1 / (2 + self._path_distance(chunk.metadata['filename'], datapoint.completion_file['filename']))
            for chunk in chunks
        ]
