import fnmatch
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import get_dataset_config_names, load_dataset, Dataset, get_dataset_infos, load_dataset_builder
from huggingface_hub import HfApi, HfFileSystem


@dataclass
class DatasetIterOut:
    chunk_num: int
    ds: Dataset


@dataclass
class DatapointIterOut:
    chunk_num: int
    dp: dict


class DataLoaderComposed:
    def __init__(self,
                 hf_path: str = 'JetBrains-Research/context-py-train',
                 hf_config: str | None = None,
                 cache_dir: str = '/mnt/data2/shared-data/lca/hf_cache/'
                 ):
        self.hf_path = hf_path
        self.cache_dir = cache_dir
        self.config_names = get_dataset_config_names(hf_path)
        # self.ds_infos = get_dataset_infos(hf_path)

        if hf_config is None:
            self.hf_config = self.config_names[0]
            # TODO: log choice of config
        elif hf_config not in self.config_names:
            raise ValueError(f'`hf_config` must be one of {self.config_names},\nYour config is {hf_config}')
        else:
            self.hf_config = hf_config
        self.data_files = self._get_data_files()
        # self.ds_builder = load_dataset_builder(self.hf_path, self.hf_config)

    def _get_data_files(self) -> list[str]:
        hf_api = HfApi()
        ds_info = hf_api.dataset_info(self.hf_path)
        filename_patterns = list()
        list_of_files = list()
        for ds_config in ds_info.card_data.get('configs'):
            if ds_config['config_name'] == self.hf_config:
                filename_patterns = [ds_files['path'] for ds_files in ds_config['data_files']]
        fs = HfFileSystem()
        for fptrn in filename_patterns:
            list_of_files.extend(fs.glob(f"datasets/{self.hf_path}/{fptrn}", detail=False))
        ds_filenames = list()
        for filename in list_of_files:
            if any(fnmatch.fnmatch(filename, f"datasets/{self.hf_path}/{pattern}") for pattern in filename_patterns):
                ds_filenames.append(filename[len(f"datasets/{self.hf_path}/"):])
        return ds_filenames

    @staticmethod
    def make_list_chunks(list_: list[Any], chunk_size: int) -> list[list[Any]]:
        chunk_number = len(list_) // chunk_size + int(len(list_) % chunk_size > 0)
        chunked_list = [list_[chunk_size * i:chunk_size * (i + 1)] for i in range(chunk_number)]
        return chunked_list

    def get_dataset(self) -> Dataset:
        pass

    def get_dataset_iterator(self, chunk_size: int | None = None) -> Iterator[DatasetIterOut]:
        pass

    def get_datapoint_iterator(self) -> Iterator[DatapointIterOut]:
        pass


class DataLoaderTrainComposed(DataLoaderComposed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dataset_iterator(self, chunk_size: int | None = None) -> Iterator[DatasetIterOut]:
        _loading_kwargs = {'split': 'train', 'cache_dir': self.cache_dir}
        if chunk_size is None:
            ds = load_dataset(self.hf_path, self.hf_config, **_loading_kwargs)
            yield DatasetIterOut(chunk_num=0, ds=ds)
        else:
            for chunk_idx, chunk_files in enumerate(self._get_filename_chunks(chunk_size)):
                ds = load_dataset(self.hf_path, self.hf_config, data_files=chunk_files, **_loading_kwargs)
                yield DatasetIterOut(chunk_num=chunk_idx, ds=ds)

    def get_datapoint_iterator(self, chunk_size: int | None = None) -> Iterator[DatapointIterOut]:
        for ds_out in self.get_dataset_iterator(chunk_size):
            for dp in ds_out.ds:
                yield DatapointIterOut(chunk_num=ds_out.chunk_num, dp=dp)

    def _get_filename_chunks(self, chunk_size: int) -> list[list[str]]:
        filenames = self.data_files
        chunks = self.make_list_chunks(filenames, chunk_size)
        return chunks
