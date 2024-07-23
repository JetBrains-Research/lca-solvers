from pipeline.data.composers.chunking_mixins import Chunk
from pipeline.data.datapoint import Datapoint

from typing import Sequence


class HarvesterMixin:  # TODO: better naming?
    # abstraction bottleneck
    chunks_sep: str
    path_comment_template: str

    def harvest(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> str:
        return self.chunks_sep.join(
            self.path_comment_template.format(filename=chunk.metadata['filename'], content=chunk.content)
            for chunk in chunks
        )
