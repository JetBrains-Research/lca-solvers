import os

import yaml
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm

from data_processing.context_composers.context_composer_path_distance import ContextComposerPathDistance
from data_processing.data_classes.datapoint_base import DatapointBase
from data_processing.data_classes.datapoint_py import DatapointPy
from data_processing.data_filters.repo_snapshot_filter_stack import SnapshotFilterStack
from data_processing.data_loading.raw_train_dataset_loading import DataLoaderTrainRaw


def get_datasets() -> dict[str, str]:
    dataset_paths = {
        'train': 'jenyag/lca-codegen-train',
    }
    return dataset_paths


def get_composer_modes() -> list[str]:
    modes = ['relevant', 'non_relevant', 'all']
    return modes


def get_hf_config(config_name: str, data_files: str, split: str = 'train') -> dict:
    cfg = {
        'config_name': config_name,
        'data_files': [
            {
                'split': split,
                'path': data_files
            },
        ]
    }
    return cfg

def compose_datapoint(source_dp: DatapointBase, context: str, context_strategy: str = 'path_distance') -> dict:
    return {
        'repo': source_dp.repo,
        'commit_hash': source_dp.commit,
        'context': context,
        'completion_file': source_dp.get_completion_file(),
        'context_strategy': context_strategy,
    }


def main() -> None:
    hf_config = {'configs': list()}

    datasets_paths = get_datasets()
    ds_name = 'train'

    context_strategy = 'path_distance'
    loader = DataLoaderTrainRaw()
    composer = ContextComposerPathDistance()
    filt = SnapshotFilterStack()

    dp_idx = 0
    chunk_idx = 0
    for ds_chunk in loader.get_dataset_iterator(16):
        hf_config = {'configs': list()}
        ds_chunk = [DatapointPy.from_hf_datapoint(hf_dp, filtering=filt) for hf_dp in
                    tqdm(ds_chunk, desc='dataset filtration')]
        api = HfApi()
        for mode in tqdm(get_composer_modes()):
            context_ds_list = list()
            for dp in ds_chunk:
                # dp = DatapointPy.from_hf_datapoint(hf_dp, filtering=filt)
                context = composer.compose_context(dp, mode=mode, max_length=1_000_000)
                context_ds_list.append(compose_datapoint(dp, context, context_strategy=context_strategy))
            context_ds = Dataset.from_list(context_ds_list)
            # context_ds.push_to_hub('jenyag/context-py-eval')
            save_path = 'train_data/tmp.parquet'
            context_ds.to_parquet(save_path)
            hf_data_dir = f'data/{ds_name}/{context_strategy}/{mode}'
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=f'{hf_data_dir}/chunk_{chunk_idx:04d}.parquet',
                repo_id='jenyag/context-py-train',
                repo_type="dataset",
            )
            os.remove(save_path)
            curr_config = get_hf_config(f'{context_strategy}_{mode}', f'{hf_data_dir}/*')
            hf_config['configs'].append(curr_config)
            # hf_config['configs'] = list(set(hf_config['configs']))
        chunk_idx += 1
    with open('train_data/hf_configs.txt', 'w') as file:
        file.write(yaml.dump(hf_config))


if __name__ == '__main__':
    main()
