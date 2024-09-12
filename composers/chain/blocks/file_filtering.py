from composers.chain.chain import File, ComposerBlock
from composers.data.datapoint import Datapoint

import random
from abc import ABC
from typing import Sequence, Type

from transformers import PreTrainedTokenizerBase


class FileFilter(ComposerBlock, ABC):
    first_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from composers.chain.blocks.file_chunking import FileChunker
        from composers.chain.blocks.file_preprocessing import FilePreprocessor
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
    requires_tokenizer = True

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
    requires_tokenizer = True

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 min_ratio: float,
                 max_ratio: float,
                 subsequence_len: int,
                 random_seed: int | None,
                 ) -> None:
        self.tokenizer = tokenizer
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.subsequence_len = subsequence_len
        self.generator = random.Random(random_seed)

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        filtered_files = list()

        for file in files:
            if len(file.content) <= self.subsequence_len:
                subsequence = file.content
            else:
                # N.B. this algorithm does NOT preserve the uniformity of token sampling
                # only the uniformity of subsequences
                start_idx = self.generator.randrange(len(file.content) - self.subsequence_len + 1)
                subsequence = file.content[start_idx:start_idx + self.subsequence_len]

            tokenized_subsequence = self.tokenizer(subsequence, return_attention_mask=False).input_ids
            ratio = len(subsequence) / len(tokenized_subsequence)

            if self.min_ratio <= ratio <= self.max_ratio:
                filtered_files.append(file)

        return filtered_files