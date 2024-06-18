from datasets import load_dataset

from lca_eval_harness.data_classes.datapoint_base import DataPointBase


class HFDataLoader:
    def __init__(self, hf_dataset_path: str, hf_split: str = 'train'):
        self.hf_ds_split = hf_split
        self.hf_ds_path = hf_dataset_path
        self.data = self._get_data()

    def _get_data(self):
        ds = load_dataset(self.hf_ds_path, split=self.hf_ds_split)
        return [DataPointBase(**dp_dict) for dp_dict in ds]



if __name__ == '__main__':
    data_loader = HFDataLoader(hf_dataset_path='ekaterina-blatova-jb/val_dataset_lora')
    print(data_loader.data)
