from pipeline.data.composers.base_composers import RankingComposer
from pipeline.data.datapoint import Datapoint

import os
import warnings
from typing import Iterable


class PathDistanceComposer(RankingComposer):  # TODO: test
    @staticmethod
    def _path_distance(path_from: str, path_to: str) -> int:
        path_from = os.path.normpath(path_from)
        path_to = os.path.normpath(path_to)

        if path_from == path_to:  # TODO: hardcode one exception in train dataset
            warnings.warn(f'Data leak: the {path_from} completion file is contained in the repo snapshot.')

        divided_path_from = path_from.split(os.path.sep)
        divided_path_to = path_to.split(os.path.sep)

        common_len = 0
        for segment_from, segment_to in zip(divided_path_from, divided_path_to):
            if segment_from == segment_to:
                common_len += 1
            else:
                break

        n_residuals_from = len(divided_path_from) - common_len - 1
        n_residuals_to = len(divided_path_to) - common_len - 1

        return n_residuals_from + n_residuals_to

    def ranking_function(self, _chunks: Iterable[str], datapoint: Datapoint) -> Iterable[int | float]:
        return map(
            lambda x: -self._path_distance(x, datapoint.completion_file['filename']),
            datapoint.repo_snapshot['filename'],
        )
