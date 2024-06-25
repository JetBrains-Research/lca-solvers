from typing import Any

from datasets import load_dataset, Dataset
from huggingface_hub import HfFileSystem
from tqdm.auto import tqdm

import os


class DataLoaderTrainRaw:
    def __init__(self, *, hf_path: str = None, cache_dir: str = None) -> None:
        if hf_path is None:
            pass  # TODO
        if cache_dir is None:
            pass  # TODO

        print(os.listdir())

        self.hf_path = hf_path
        self.cache_dir = cache_dir

    def load_data_file(self, hf_files: str | list[str]) -> Dataset:
        return load_dataset(
            self.hf_path,
            data_dir='train',
            data_files=hf_files,
            split='train',
            cache_dir=self.cache_dir,
        )

    @staticmethod
    def make_list_chunks(list_: list[Any], chunk_size: int) -> list[list[Any]]:
        chunk_number = len(list_) // chunk_size + int(len(list_) % chunk_size > 0)
        chunked_list = [list_[chunk_size * i:chunk_size * (i + 1)] for i in range(chunk_number)]
        return chunked_list

    def get_dataset_iterator(self, chunk_size: int = 16):
        fs = HfFileSystem()

        list_of_files = fs.ls(f'datasets/{self.hf_path}/train', detail=False)
        list_of_filenames = [fp.split(fs.sep)[-1] for fp in list_of_files]

        list_of_filenames_by_chunks = self.make_list_chunks(list_of_filenames, chunk_size=chunk_size)

        for ds_filenames in tqdm(list_of_filenames_by_chunks, desc='Chunks of the Dataset'):
            ds = self.load_data_file(ds_filenames)
            yield ds

    def get_dp_iterator(self, chunk_size: int = 16):
        for ds in self.get_dataset_iterator(chunk_size=chunk_size):
            for dp in ds:
                yield dp
