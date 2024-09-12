from composers.chain.utils import ReprMixin
from composers.data.datapoint import Datapoint

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class File:
    content: str
    metadata: dict[str, Any]


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any]
    file_ref: File
    rank: list = field(default_factory=list)  # of comparable elements


BlockArgs = Sequence[File] | Sequence[Chunk]


class ComposerBlock(ABC, ReprMixin):
    first_block_permit: bool = False
    last_block_permit: bool = False
    requires_tokenizer: bool = False

    @property
    @abstractmethod
    def next_blocks(self) -> tuple[type, ...]:
        raise NotImplementedError

    def check_next_block(self, block) -> None:
        if not isinstance(block, self.next_blocks):
            raise ValueError(f'{type(block).__name__} cannot be used after {type(self).__name__}.')

    @abstractmethod
    def __call__(self, args: BlockArgs, datapoint: Datapoint) -> BlockArgs | str:
        raise NotImplementedError


class UnsafeComposerChain:
    def __init__(self, *blocks: ComposerBlock) -> None:
        self.blocks = blocks

    def __call__(self, datapoint: Datapoint) -> Any:
        x = [
            File(content=cnt, metadata={'filename': fn})
            for fn, cnt in zip(*datapoint.repo_snapshot.values())
        ]
        for block in self.blocks:
            x = block(x, datapoint)
        return x


class ComposerChain(UnsafeComposerChain):
    def __init__(self, *blocks: ComposerBlock) -> None:
        if not blocks:
            raise ValueError('ComposerChain instance must contain at least one element.')
        elif not blocks[0].first_block_permit:
            raise ValueError(f'{type(blocks[0]).__name__} cannot start a chain of blocks.')
        elif not blocks[-1].last_block_permit:
            raise ValueError(f'{type(blocks[-1]).__name__} cannot end a chain of blocks.')

        for block, next_block in zip(blocks[:-1], blocks[1:]):
            block.check_next_block(next_block)

        super().__init__(*blocks)
