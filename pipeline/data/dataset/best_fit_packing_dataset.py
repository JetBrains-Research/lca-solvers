"""
Implementation of the Fewer Truncations Improve Language Modeling paper.
Reference: https://arxiv.org/abs/2404.10830

What is NOT implemented:
  1. path_comment_template support
  2. path_comment-free packing
  3. composers integration
  4. loss decomposition
"""
from pipeline.data.categories import UNDEFINED_CATEGORY_ID
from pipeline.data.dataset.segment_tree import SegmentTree
from pipeline.data.preprocessors.preprocessor_base import BatchMetadata, PreprocessedBatch

import copy
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, TypedDict

import torch
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase


class DocumentTokenDistribution(TypedDict):
    path_comment: int
    content: list[int]


@dataclass
class BFPChunk:
    document_id: str
    start_line_idx: int
    end_line_idx: int


class BestFitPackingDataset(Dataset):
    def __init__(self,
                 hf_dataset: HuggingFaceDataset,
                 tokenizer: PreTrainedTokenizerBase,
                 max_seq_len: int,
                 delimiter: str,
                 token_dist: dict[str, DocumentTokenDistribution] | None,
                 cache_dir: str,
                 verbose: bool,
                 random_seed: int | None,
                 ) -> None:
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        self.delimiter_len = len(tokenizer(delimiter, add_special_tokens=False).input_ids)
        self.verbose = verbose
        self.random_seed = random_seed

        if token_dist is None:
            path = os.path.join(cache_dir, 'token_dist.json')

            if os.path.exists(path):
                with open(path) as stream:
                    token_dist = json.load(stream)
            else:
                token_dist = self._calc_token_dist()
                with open(path, 'w') as stream:
                    json.dump(token_dist, stream)
        else:
            token_dist = copy.deepcopy(token_dist)

        # count sort buffer
        self.chunk_buffer: list[list[BFPChunk]] = [[] for _ in range(max_seq_len + 1)]
        self._fill_chunk_buffer(token_dist)
        self.bin_to_items = self._pack_bins()

        # full shuffle support
        self.bin_permutations = dict()
        self.bin_item_permutations = dict()
        self.chunk_permutations = dict()

        self.document_ids = hf_dataset['datapoint_identifier']

    def _calc_token_dist(self) -> dict[str, DocumentTokenDistribution]:
        token_dist = dict()

        ds_iter = tqdm(
            iterable=self.hf_dataset,
            desc='Precalculating token distribution',
            disable=not self.verbose,
        )

        for document in ds_iter:
            document_id = document['datapoint_identifier']
            if document_id in token_dist:
                raise RuntimeError('The original dataset contains repeated '
                                   f'identifiers, e.g. {document_id}.')

            path_comment = f'# {document["filename"]}\n'
            lines = [line + '\n' for line in document['content'].rstrip().split('\n')]

            path_comment_len, *lines_len = list(map(len, self.tokenizer(
                text=[path_comment] + lines,
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids))

            token_dist[document_id] = DocumentTokenDistribution(
                path_comment=path_comment_len,
                content=lines_len
            )

        return token_dist

    def _split_document(self, doc_tok_dist: DocumentTokenDistribution) -> Iterator:
        path_comment_len = doc_tok_dist['path_comment']
        lines_len = doc_tok_dist['content']

        chunk_len = path_comment_len
        start_line_idx = 0

        for line_idx, line_len in enumerate(lines_len):
            if chunk_len + line_len > self.max_seq_len:
                yield chunk_len, start_line_idx, line_idx
                chunk_len = path_comment_len
                start_line_idx = line_idx
            else:
                chunk_len += line_len

        yield chunk_len, start_line_idx, len(lines_len)

    def _fill_chunk_buffer(self, token_dist: dict[str, DocumentTokenDistribution]) -> None:
        for doc_id, tok_dist in token_dist.items():
            for chunk_len, start_line_idx, end_line_idx in self._split_document(tok_dist):
                self.chunk_buffer[chunk_len].append(BFPChunk(
                    document_id=doc_id,
                    start_line_idx=start_line_idx,
                    end_line_idx=end_line_idx,
                ))

    def _pack_bins(self) -> list[list[tuple[int, int]]]:
        segment_tree = SegmentTree(self.max_seq_len)
        bin_to_items = defaultdict(list)
        space_to_bins = [[] for _ in range(self.max_seq_len + 1)]
        space_to_bins[-1] = list(range(len(self.hf_dataset)))

        chunk_buffer_iter = tqdm(
            iterable=list(enumerate(self.chunk_buffer))[::-1],
            desc='Packing chunks',
            disable=not self.verbose,
        )

        for chunk_len, chunks in chunk_buffer_iter:
            for chunk_id, _ in enumerate(chunks):
                weight = chunk_len + self.delimiter_len
                best_fit_capacity = segment_tree.find_best_fit_capacity(weight)

                bin_idx = space_to_bins[best_fit_capacity].pop(0)
                is_bin_empty = not bin_to_items[bin_idx]
                remaining_space = best_fit_capacity - weight + is_bin_empty

                if not space_to_bins[best_fit_capacity]:
                    segment_tree.delete(best_fit_capacity)

                if not segment_tree[remaining_space]:
                    segment_tree.insert(remaining_space)
                space_to_bins[remaining_space].append(bin_idx)

                bin_to_items[bin_idx].append((chunk_len, chunk_id))

        return list(bin_to_items.values())

    def __len__(self) -> int:
        return len(self.bin_to_items)

    def __getitem__(self, sample_idx: int) -> PreprocessedBatch:
        epoch_idx = sample_idx // len(self)
        bin_idx = sample_idx % len(self)

        if epoch_idx not in self.bin_permutations:
            if self.random_seed is not None:
                generator = random.Random(self.random_seed + epoch_idx)
            else:
                generator = random.Random()

            self.bin_permutations[epoch_idx] = generator.sample(range(len(self)), len(self))
            self.bin_item_permutations[epoch_idx] = [
                generator.sample(range(len(items)), len(items))
                for items in self.bin_to_items
            ]
            self.chunk_permutations[epoch_idx] = [
                generator.sample(range(len(chunks)), len(chunks))
                for chunks in self.chunk_buffer
            ]

        bin_idx = self.bin_permutations[epoch_idx][bin_idx]
        item_ids = self.bin_to_items[bin_idx]
        item_ids = [item_ids[i] for i in self.bin_item_permutations[epoch_idx][bin_idx]]
        chunks = [
            self.chunk_buffer[chunk_len][self.chunk_permutations[epoch_idx][chunk_len][chunk_id]]
            for chunk_len, chunk_id in item_ids
        ]

        chunked_text = list()
        for chunk in chunks:
            datapoint_idx = self.document_ids.index(chunk.document_id)
            document = self.hf_dataset[datapoint_idx]

            path_comment = f'# {document["filename"]}\n'
            lines = document['content'].rstrip().split('\n')
            lines = lines[chunk.start_line_idx:chunk.end_line_idx]

            content = path_comment + '\n'.join(lines) + '\n'
            chunked_text.append(content)

        tokenized_sample = self.tokenizer(
            text=self.delimiter.join(chunked_text),
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt')
        seq_len = tokenized_sample.input_ids.shape[-1] - 1

        return PreprocessedBatch(
            input_ids=tokenized_sample.input_ids[0, :-1],
            target_ids=tokenized_sample.input_ids[0, 1:],
            loss_mask=torch.ones(seq_len, dtype=torch.bool),
            completion_mask=torch.ones(seq_len, dtype=torch.bool),
            category_ids=torch.full((seq_len,), UNDEFINED_CATEGORY_ID),
            input_attn_mask=tokenized_sample.attention_mask[0, :-1],
            target_attn_mask=tokenized_sample.attention_mask[0, 1:].bool(),
            metadata=BatchMetadata(),
        )
