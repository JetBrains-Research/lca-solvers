from pipeline.configs.composer_config import ComposerConfig
from pipeline.data.composed_datapoint import ComposedDatapoint
from pipeline.data.datapoint import Datapoint

import abc
from typing import Any, Iterable

from datasets import Dataset


class ComposerBase(abc.ABC):
    def __init__(self, config: ComposerConfig) -> None:
        self.config = config

    def get_pre_context_prompt(self, datapoint: Datapoint) -> str:
        return self.config.pre_context_prompt.format(datapoint.repo)

    def get_post_context_prompt(self, _datapoint: Datapoint) -> str:
        return self.config.post_context_prompt

    @abc.abstractmethod
    def compose_context(self, datapoint: Datapoint) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def compose_completion(self, datapoint: Datapoint) -> str:
        raise NotImplementedError

    def compose(self, datapoint: dict[str, Any]) -> ComposedDatapoint:
        datapoint = Datapoint(**datapoint)
        return ComposedDatapoint(
            pre_context_prompt=self.get_pre_context_prompt(datapoint),
            composed_context=self.compose_context(datapoint) + self.get_post_context_prompt(datapoint),
            composed_completion=self.compose_completion(datapoint),
        )

    def compose_dataset(self,
                        dataset: Dataset,
                        writer_batch_size: int = 100,
                        num_proc: int = 4,
                        **map_kwargs,
                        ) -> Dataset:
        return dataset.map(
            function=self.compose,
            remove_columns=dataset.column_names,
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
            desc=f'Applying {type(self).__name__} to a given dataset',
            **map_kwargs,
        )


class GrainedComposer(ComposerBase):
    @abc.abstractmethod
    def chunk_datapoint(self, datapoint: Datapoint) -> Iterable[str]:
        raise NotImplementedError

    def combine_chunks(self, chunks: Iterable[str]) -> str:
        return self.config.chunks_sep.join(chunks)


class RankingComposer(GrainedComposer):
    @abc.abstractmethod
    def ranking_function(self, chunks: Iterable[str], datapoint: Datapoint) -> Iterable[int | float]:
        raise NotImplementedError

    def compose_context(self, datapoint: Datapoint) -> str:
        chunks = self.chunk_datapoint(datapoint)
        ranks = self.ranking_function(chunks, datapoint)
        chunks = [chunk for _, chunk in sorted(zip(ranks, chunks), key=lambda x: x[0])]
        return self.combine_chunks(chunks)
