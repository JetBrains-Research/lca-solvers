from pipeline.data.composers.chain import ComposerBlock, ComposerChain
from pipeline.data.composers.composer_base import ComposerBase
from pipeline.data.composers.utils import ReprMixin
from pipeline.data.datapoint import Datapoint

from typing import Sequence


class ChainedComposer(ComposerBase, ComposerChain, ReprMixin):
    def __init__(self, blocks: Sequence[ComposerBlock], *args, **kwargs) -> None:
        ComposerBase.__init__(self, *args, **kwargs)
        ComposerChain.__init__(self, *blocks)

    def compose_context(self, datapoint: Datapoint) -> str:
        return self.__call__(datapoint)
