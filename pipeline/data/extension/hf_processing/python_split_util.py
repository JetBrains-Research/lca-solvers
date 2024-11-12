from datasets import Dataset

from pipeline.data.extension.hf_processing.data_classes.hf_completion_file_local import HFCompletionFileLocal
from pipeline.data.extension.scripts.extract_lca_test_repos import extract_lca_test_data


def define_split(dp, test_data_list: list[tuple[str, str, str]]) -> str:
    test_identifiers = [HFCompletionFileLocal.generate_unique_identifier(*test_data) for test_data in test_data_list]
    test_repos = [test_data[0] for test_data in test_data_list]
    dp_identifier = HFCompletionFileLocal.generate_unique_identifier(dp['repo'], dp['filename'], dp['commit_hash'])
    if dp_identifier in test_identifiers:
        return 'test'
    elif dp['repo'] in test_repos:
        if dp['year'] >= 2022:
            return 'val'
        else:
            return 'val_old'
    else:
        return 'train'


def split_python_dataset(ds: Dataset, num_proc: int = 16):
    test_data_dict = extract_lca_test_data()
    test_data_list = list()
    for config_test_data in test_data_dict.values():
        test_data_list.extend(config_test_data)
    split_names = ['train', 'val', 'val_old', 'test']
    result = dict()
    updated_ds = ds.map(lambda dp: {'split_name': define_split(dp, test_data_list)}, num_proc=num_proc)
    for split_name in split_names:
        split_ds = updated_ds.filter(lambda dp: dp['split_name']==split_name).remove_columns('split_name')
        result[split_name] = split_ds
    return result

if __name__ == '__main__':
    pass
