from pipeline.configs.composer_config import ComposerConfig
from pipeline.data.datapoint import Datapoint

import abc
from typing import Iterable


class ContextComposer(abc.ABC):
    @abc.abstractmethod
    def compose_context(self, datapoint: Datapoint) -> str:
        raise NotImplementedError


class GrainedComposer(ContextComposer):
    def __init__(self, config: ComposerConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def chunk_datapoint(self, datapoint: Datapoint) -> Iterable[str]:
        raise NotImplementedError

    def combine_chunks(self, chunks: Iterable[str]) -> str:
        return (self.config.pre_context_prompt +
                self.config.chunks_sep.join(chunks) +
                self.config.post_context_prompt)


class RankingComposer(GrainedComposer):
    @abc.abstractmethod
    def ranking_function(self,
                         chunks: Iterable[str],
                         datapoint: Datapoint | None = None,
                         ) -> Iterable[int | float]:
        raise NotImplementedError

    def compose_context(self, datapoint: Datapoint) -> str:
        chunks = self.chunk_datapoint(datapoint)
        ranks = self.ranking_function(chunks, datapoint)
        chunks = [chunk for _, chunk in sorted(zip(ranks, chunks), key=lambda x: -x[0])]

        context = self.combine_chunks(chunks)
        context += self.config.path_comment_template.format(**datapoint.completion_file)

        return context
