from pipeline.data.composers.chain import File, Chunk, ComposerBlock
from pipeline.data.datapoint import Datapoint

from abc import ABC
from enum import Enum
from typing import NamedTuple, Sequence, TypeVar, Type

import tree_sitter
import tree_sitter_python

T = TypeVar('T')


class FileChunker(ComposerBlock, ABC):
    first_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from pipeline.data.composers.blocks.chunk_harvesting import ChunkHarvester
        from pipeline.data.composers.blocks.chunk_ranking import ChunkRanker
        return ChunkRanker, ChunkHarvester


class FileGrainedChunker(FileChunker):  # identity chunker
    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[Chunk]:
        return [Chunk(content=file.content, metadata=file.metadata, file_ref=file) for file in files
                # TODO: remove temporary hardcoded solution for data leakage
                if file.metadata['filename'] != 'tinygrad/llops/ops_llvm.py']


class CodeSegment(str, Enum):
    COMMENT = 'comment_segment'
    DOCSTRING = 'docstring_segment'
    IMPORT = 'import_segment'
    CODE = 'code_segment'
    UNDEFINED = 'undefined_segment'

    @classmethod
    def from_node(cls: Type[T], node: tree_sitter.Node) -> T:
        if 'comment' in node.type.lower():
            return cls.COMMENT
        elif node.type == 'string' and node.text.startswith(CodeSegmentGrainedChunker.DOCSTRING_PREFIX):
            return cls.DOCSTRING
        elif 'import' in node.type.lower():
            return cls.IMPORT
        elif node.child_count == 0:
            return cls.CODE
        else:
            return cls.UNDEFINED


class Segment(NamedTuple):
    start_byte: int
    type: CodeSegment


class CodeSegmentGrainedChunker(FileChunker):
    ENCODING = 'utf8'
    DOCSTRING_PREFIX = (bytes("'''", ENCODING), bytes('"""', ENCODING))

    def __init__(self) -> None:
        py_language = tree_sitter.Language(tree_sitter_python.language())
        self.parser = tree_sitter.Parser(py_language)

    def dfs_segmentation(self, segments: list[Segment], node: tree_sitter.Node) -> None:
        segment_type = CodeSegment.from_node(node)

        if segment_type != CodeSegment.UNDEFINED:
            if len(segments) == 0 or segment_type != segments[-1].type:
                segments.append(Segment(node.start_byte, segment_type))
        else:
            for child in node.children:
                self.dfs_segmentation(segments, child)

    @staticmethod
    def remove_leading_whitespaces(string: str) -> str:
        return '\n'.join(map(str.lstrip, string.splitlines()))

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[Chunk]:
        chunks = list()

        for file in files:
            if not file.metadata['filename'].endswith('.py'):
                chunks.append(Chunk(
                    content=file.content,
                    metadata=file.metadata | {'segment_type': CodeSegment.UNDEFINED},
                    file_ref=file,
                ) for file in files)
            # TODO: remove temporary hardcoded solution for data leakage
            if file.metadata['filename'] == 'tinygrad/llops/ops_llvm.py':
                continue

            segments = list()
            bytecode = bytes(file.content, self.ENCODING)
            tree = self.parser.parse(bytecode)
            self.dfs_segmentation(segments, tree.root_node)

            dummy_segment = Segment(len(bytecode), CodeSegment.UNDEFINED)
            segments.append(dummy_segment)

            comments_chunk = Chunk(
                content='', metadata=file.metadata | {'segment_type': CodeSegment.COMMENT}, file_ref=file)
            docstrings_chunk = Chunk(
                content='', metadata=file.metadata | {'segment_type': CodeSegment.DOCSTRING}, file_ref=file)
            imports_chunk = Chunk(
                content='', metadata=file.metadata | {'segment_type': CodeSegment.IMPORT}, file_ref=file)
            code_chunk = Chunk(
                content='', metadata=file.metadata | {'segment_type': CodeSegment.CODE}, file_ref=file)

            for i in range(len(segments) - 1):
                start = segments[i].start_byte
                end = segments[i + 1].start_byte
                segment_str = bytecode[start:end].decode(self.ENCODING)
                segment_type = segments[i].type

                match segment_type:
                    case CodeSegment.COMMENT:
                        if segment_str.count('\n') >= 2:
                            comments_chunk.content += self.remove_leading_whitespaces(segment_str)
                    case CodeSegment.DOCSTRING:
                        docstrings_chunk.content += self.remove_leading_whitespaces(segment_str)
                    case CodeSegment.IMPORT:
                        imports_chunk.content += segment_str
                    case CodeSegment.CODE:
                        code_chunk.content += segment_str
                    case _:
                        raise RuntimeError  # indicates a bug

            for chunk in (comments_chunk, docstrings_chunk, imports_chunk, code_chunk):
                if chunk.content.strip():
                    chunk.content = chunk.content.rstrip()
                    chunks.append(chunk)

        return chunks