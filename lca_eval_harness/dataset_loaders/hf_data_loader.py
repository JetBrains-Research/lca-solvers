from datasets import load_dataset

from lca_eval_harness.data_classes.datapoint_base import DataPointBase
from lca_eval_harness.data_classes.datapoint_prefix_indexes import DataPointPrefixIndexes
from lca_eval_harness.dataset_loaders.base_data_loader import BaseDataLoader


class HFDataLoader(BaseDataLoader):
    def __init__(self, hf_dataset_path: str, hf_split: str = 'train'):
        self.hf_ds_split = hf_split
        self.hf_ds_path = hf_dataset_path
        self._data = self._get_data()

    @property
    def data(self):
        return self._data

    def _get_data(self) -> list[DataPointBase]:
        ds = load_dataset(self.hf_ds_path, split=self.hf_ds_split)
        data = list()
        for dp_dict in ds:
            if 'split_idxs' in dp_dict:
                split_idxs = dp_dict.pop('split_idxs', None)
                dp = DataPointPrefixIndexes(prefix_split_indexes=split_idxs, **dp_dict)
            else:
                dp = DataPointBase(**dp_dict)
            data.append(dp)
        return data


if __name__ == '__main__':
    data_loader = HFDataLoader(hf_dataset_path='ekaterina-blatova-jb/val_dataset_lora')
    print(data_loader.data)
