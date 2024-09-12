from composers.chain.chain import File, ComposerBlock
from composers.data.datapoint import Datapoint

from abc import ABC
from typing import Sequence, Type

import tree_sitter
import tree_sitter_python
import warnings


class FilePreprocessor(ComposerBlock, ABC):
    first_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from composers.chain.blocks.file_chunking import FileChunker
        from composers.chain.blocks.file_filtering import FileFilter
        return FileFilter, FilePreprocessor, FileChunker


class EmptyLinesRemovalPreprocessor(FilePreprocessor):
    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        for file in files:
            file.content = '\n'.join(line for line in file.content.split('\n') if line.strip())
        return files


class DeclarationOnlyPreprocessor(FilePreprocessor):
    ENCODING = 'utf8'

    def __init__(self) -> None:
        py_language = tree_sitter.Language(tree_sitter_python.language())
        self.parser = tree_sitter.Parser(py_language)

    def __call__(self, files: Sequence[File], datapoint: Datapoint) -> Sequence[File]:
        for file in files:
            if not file.metadata['filename'].endswith('.py'):
                continue

            bytecode = bytes(file.content, self.ENCODING)
            queue = [self.parser.parse(bytecode).root_node]
            declarations = list()

            while queue:
                node = queue.pop()
                queue.extend(reversed(node.children))

                if node.type not in ('function_definition', 'class_definition'):
                    continue

                start = bytecode[:node.start_byte].rfind(b'\n') + 1
                for child in node.children:
                    if child.type == ':':
                        end = child.end_byte
                        break
                else:
                    warnings.warn(f'A corrupted {file.metadata["filename"]} file structure '
                                  f'has been detected in the {datapoint.repo} repository.')
                    end = node.end_byte

                declaration = bytecode[start:end]
                declaration = declaration.decode('utf8') + ' ...'
                declarations.append(declaration)

            file.content = '\n'.join(declarations)
        return files
