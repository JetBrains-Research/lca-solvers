from dataclasses import dataclass


@dataclass
class DataPointBase:
    completion_file: str
    context: str
    repo: str
    commit_hash: str
    context_strategy: str

    def split_completion_file(self) -> (list[str], list[str]):
        raise NotImplementedError


if __name__ == '__main__':
    from datasets import load_dataset

    ds = load_dataset('ekaterina-blatova-jb/val_dataset_lora', split='train')
    datapoints = list()
    for dp_dict in ds:
        datapoints.append(DataPointBase(**dp_dict))
    print(len(datapoints))
    print(datapoints[0].repo, datapoints[0].commit_hash, datapoints[0].context_strategy)
