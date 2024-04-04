import os
import random
from functools import partial
from typing import Tuple

from context_composers.context_composer_base import ContextComposerBase
from data_classes.datapoint_base import DatapointBase


class ContextComposerPathDistance(ContextComposerBase):
    def __init__(self):
        super().__init__()
        pass

    def compose_context(self, datapoint: DatapointBase, mode: str | None = None, max_length: int = -1) -> str:
        # relevant_context = datapoint.get_relevant_context()
        # non_relevant_context = datapoint.get_non_relevant_context()
        compl_filename = datapoint.get_completion_filenames()[0]

        if mode is None:
            context_dict = dict()
        elif mode == 'relevant':
            context_dict = datapoint.get_relevant_context()
        elif mode == 'non_relevant':
            context_dict = datapoint.get_non_relevant_context()
        elif mode == 'all':
            context_dict = datapoint.get_relevant_context() | datapoint.get_non_relevant_context()
        else:
            raise ValueError("The mode must be 'relevant', 'non_relevant' or 'all'")
        paths = self._sort_filepaths(compl_filename, list(context_dict))
        context = ''
        for filepath in paths:
            content = context_dict[filepath]
            context += f'# {filepath}\n\n{content}\n\n'
        context += f'# {compl_filename}'
        if max_length > 0:
            context = context[-max_length:]

        return context

    @staticmethod
    def _path_distance(path_from: str, path_to: str) -> tuple[int, int, int]:
        divided_path_from = os.path.normpath(path_from).split(os.path.sep)
        divided_path_to = os.path.normpath(path_to).split(os.path.sep)
        common_len = 0
        for el1, el2 in zip(divided_path_from, divided_path_to):
            if el1 == el2:
                common_len += 1
            else:
                break
        # return len(divided_path_from) - common_len - 1
        return ((len(divided_path_from) - common_len - 1) + (len(divided_path_to) - common_len - 1),
                len(divided_path_to) - common_len - 1,
                len(divided_path_from) - common_len - 1,
                )

    def _sort_filepaths(self, path_from: str, list_of_filepathes: list[str]) -> list[str]:
        _pd = partial(self._path_distance, path_from=path_from)
        paths_by_distance = sorted(list_of_filepathes, key=lambda fn: _pd(path_to=fn), reverse=True)
        return paths_by_distance
