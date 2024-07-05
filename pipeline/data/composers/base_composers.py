from pipeline.data.composed_datapoint import ComposedDatapoint
from pipeline.data.datapoint import Datapoint

import abc
from typing import Any, Iterable

from datasets import Dataset


# TODO: cache results?
class ComposerBase(abc.ABC):
    def __init__(self,
                 pre_context_prompt: str,
                 chunks_sep: str,
                 post_context_prompt: str,
                 path_comment_template: str,
                 ) -> None:
        self.pre_context_prompt = pre_context_prompt
        self.chunks_sep = chunks_sep
        self.post_context_prompt = post_context_prompt
        self.path_comment_template = path_comment_template

    def get_pre_context_prompt(self, datapoint: Datapoint) -> str:
        return self.pre_context_prompt.format(datapoint.repo)

    def get_post_context_prompt(self, _datapoint: Datapoint) -> str:
        return self.post_context_prompt

    @abc.abstractmethod
    def compose_context(self, datapoint: Datapoint) -> str:
        raise NotImplementedError

    def compose_completion(self, datapoint: Datapoint) -> str:
        template_with_inserted_path = self.path_comment_template.format(
            filename=datapoint.completion_file['filename'],
            content='{content}',
        )

        for i, line in enumerate(template_with_inserted_path.split('\n')):
            if '{content}' in line:
                offset = i
                break
        else:
            raise RuntimeError('The path_comment_template does not contain a content field.')

        for line_category_ids in datapoint.completion_lines.values():
            for i in range(len(line_category_ids)):
                line_category_ids[i] += offset

        return self.path_comment_template.format(**datapoint.completion_file)

    def compose(self, datapoint: dict[str, Any]) -> ComposedDatapoint:
        datapoint = Datapoint(**datapoint)
        return ComposedDatapoint(
            pre_context_prompt=self.get_pre_context_prompt(datapoint),
            composed_context=self.compose_context(datapoint) + self.get_post_context_prompt(datapoint),
            composed_completion=self.compose_completion(datapoint),
            completion_lines=datapoint.completion_lines,
        )

    def compose_dataset(self,
                        dataset: Dataset,
                        writer_batch_size: int = 128,
                        num_proc: int = 4,
                        **map_kwargs,
                        ) -> Dataset:
        return dataset.map(
            function=self.compose,
            remove_columns=map_kwargs.pop('remove_columns', dataset.column_names),
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
            desc=map_kwargs.pop('desc', f'Applying {type(self).__name__} to a given dataset'),
            **map_kwargs,
        )


class GrainedComposer(ComposerBase, abc.ABC):
    def chunk_datapoint(self, datapoint: Datapoint) -> Iterable[str]:
        return [
            self.path_comment_template.format(filename=fn, content=cnt)
            for fn, cnt in zip(datapoint.repo_snapshot['filename'], datapoint.repo_snapshot['content'])
        ]

    def combine_chunks(self, chunks: Iterable[str]) -> str:
        return self.chunks_sep.join(chunks)


class RankingComposer(GrainedComposer):
    @abc.abstractmethod
    def ranking_function(self, chunks: Iterable[str], datapoint: Datapoint) -> Iterable[int | float]:
        raise NotImplementedError

    def compose_context(self, datapoint: Datapoint) -> str:
        chunks = self.chunk_datapoint(datapoint)
        ranks = self.ranking_function(chunks, datapoint)
        chunks = [chunk for _, chunk in sorted(zip(ranks, chunks), key=lambda x: x[0])]
        return self.combine_chunks(chunks)
