from pipeline.data.composers.chunking_mixins import Chunk
from pipeline.data.datapoint import Datapoint

from collections import defaultdict
from typing import Sequence


class HarvesterMixin:
    chunks_sep: str
    path_comment_template: str

    def harvest(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> str:
        return self.chunks_sep.join(
            self.path_comment_template.format(filename=chunk.metadata['filename'], content=chunk.content)
            for chunk in chunks
        )


class LinesHarvester(HarvesterMixin):
    def harvest(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> str:
        harvested_files = defaultdict(list)
        for chunk in chunks:
            harvested_files[chunk.metadata['filename']].append(chunk.content)

        return self.chunks_sep.join(
            self.path_comment_template.format(filename=fn, content='\n'.join(cnt))
            for fn, cnt in harvested_files.items()
        )
