import os

import yaml
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm

from data_processing.context_composers.context_composer_path_distance import ContextComposerPathDistance
from data_processing.data_classes.datapoint_base import DatapointBase
from data_processing.data_classes.datapoint_py import DatapointPy
from data_processing.data_filters.repo_snapshot_filter_stack import SnapshotFilterStack


def get_datasets() -> dict[str, str]:
    dataset_paths = {
        'small': 'JetBrains-Research/lca-codegen-small',
        'medium': 'JetBrains-Research/lca-codegen-medium',
        'large': 'JetBrains-Research/lca-codegen-large',
        'huge': 'JetBrains-Research/lca-codegen-huge',
    }
    return dataset_paths


def get_composer_modes() -> list[str]:
    modes = ['relevant', 'non_relevant', 'all']
    return modes


def get_hf_config(config_name: str, data_files: str) -> dict:
    cfg = {
        'config_name': config_name,
        'data_files': [
            {
                'split': 'test',
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
    print(yaml.dump(hf_config))

    datasets_paths = get_datasets()

    context_strategy = 'path_distance'
    pbar = tqdm(datasets_paths.items(), desc='Dataset')
    for ds_name, ds_path in pbar:
        pbar.set_description(desc=ds_name)
        ds = load_dataset(
                ds_path,
                split='test',

                # cache_dir=self.cache_dir,
            )
        composer = ContextComposerPathDistance()
        filt = SnapshotFilterStack()
        ds = [DatapointPy.from_hf_datapoint(hf_dp, filtering=filt) for hf_dp in tqdm(ds, desc='dataset filtration')]
        api = HfApi()

        for mode in tqdm(get_composer_modes()):
            context_ds_list = list()
            for idx, dp in enumerate(tqdm(ds)):
                # dp = DatapointPy.from_hf_datapoint(hf_dp, filtering=filt)
                context = composer.compose_context(dp, mode=mode, max_length=1_000_000)
                context_ds_list.append(compose_datapoint(dp, context, context_strategy=context_strategy))
            context_ds = Dataset.from_list(context_ds_list)
            # context_ds.push_to_hub('jenyag/context-py-eval')
            save_path = 'tmp.parquet'
            context_ds.to_parquet(save_path)
            hf_data_dir = f'data/{ds_name}/{context_strategy}/{mode}'
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=f'{hf_data_dir}/test_data.parquet',
                repo_id='jenyag/context-py-eval',
                repo_type="dataset",
            )
            os.remove(save_path)
            curr_config = get_hf_config(f'{ds_name}_{context_strategy}_{mode}', f'{hf_data_dir}/*')
            hf_config['configs'].append(curr_config)
    with open('hf_configs.txt', 'w') as file:
        file.write(yaml.dump(hf_config))
            # raise Exception
                # if idx > 2:
                #     break


if __name__ == '__main__':
    main()
