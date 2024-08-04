from pipeline.data.composers.chain import File, ComposerBlock
from pipeline.data.datapoint import Datapoint

from abc import ABC
from typing import Sequence, Type

from transformers import PreTrainedTokenizerBase


class FileFilter(ComposerBlock, ABC):
    first_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from pipeline.data.composers.blocks.file_chunking import FileChunker
        from pipeline.data.composers.blocks.file_preprocessing import FilePreprocessor
        return FileFilter, FilePreprocessor, FileChunker


class InclusiveFileExtensionFilter(FileFilter):
    def __init__(self, whitelist: list[str]) -> None:
        self.whitelist = tuple(whitelist)

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        return [file for file in files if file.metadata['filename'].endswith(self.whitelist)]


class ExclusiveFileExtensionFilter(FileFilter):
    def __init__(self, blacklist: list[str]) -> None:
        self.blacklist = tuple(blacklist)

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        return [file for file in files if not file.metadata['filename'].endswith(self.blacklist)]


class EmptyFileFilter(FileFilter):
    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        return [file for file in files if file.content.strip()]


class FileLengthFilter(FileFilter):
    def __init__(self, min_len: int, max_len: int) -> None:
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        return [file for file in files if self.min_len <= len(file.content) <= self.max_len]


class TokenizedFileLengthFilter(FileFilter):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, min_len: int, max_len: int) -> None:
        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        filtered_files = list()

        for file in files:
            if 'num_tokens' not in file.metadata:
                tokenized_file = self.tokenizer(file.content, return_attention_mask=False).input_ids
                file.metadata['num_tokens'] = len(tokenized_file)

            if self.min_len <= file.metadata['num_tokens'] <= self.max_len:
                filtered_files.append(file)

        return filtered_files


class CharTokenRatioFilter(FileFilter):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, min_ratio: float, max_ratio: float) -> None:
        self.tokenizer = tokenizer
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        filtered_files = list()

        for file in files:
            if 'num_tokens' not in file.metadata:
                tokenized_file = self.tokenizer(file.content, return_attention_mask=False).input_ids
                file.metadata['num_tokens'] = len(tokenized_file)

            if self.min_ratio <= len(file.content) / file.metadata['num_tokens'] <= self.max_ratio:
                filtered_files.append(file)

        return filtered_files