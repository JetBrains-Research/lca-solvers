import json
import shutil
from dataclasses import asdict
from pathlib import Path

import click
from datasets import Dataset
from tqdm import tqdm

from pipeline.data.extension.hf_processing.data_classes.hf_completion_file_local import HFCompletionFileLocal
from pipeline.data.extension.hf_processing.hf_features.completion_file_features import completion_file_features_local
from pipeline.data.extension.hf_processing.python_split_util import split_python_dataset
from pipeline.data.extension.lca_filesystem.data_classes.parsed_filepathes import ProcessedCompletionFilePath, CompletionFilePath, \
    CommitMetadataPath
from pipeline.data.extension.lca_filesystem.lca_filesystem import LCAFilesystem
from pipeline.data.extension.line_classification.data_classes.processed_file import ProcessedFile
from pipeline.data.extension.repo_processing.data_classes.commit_metadata import CommitMetadata
from pipeline.data.extension.repo_processing.data_classes.file_mod_metadata import FileModMetadata


def write_chunk(hf_data: list[dict], chunk_count: int,
                lca_filesystem: LCAFilesystem,
                # path_to_tmp_save: Path
                ) -> (list[dict], int):
    # print(len(hf_data), 'datapoints are written in',
    #       str(path_to_tmp_save / f'hf_data_chunk_{chunk_count :04d}.json'), end=' ')
    # with open(path_to_tmp_save / f'hf_data_chunk_{chunk_count :04d}.json', 'w') as f:
    #     json.dump(hf_data, f, indent=4)
    dataset = Dataset.from_list(hf_data, features=completion_file_features_local)

    num_proc = 16

    if 'python' in lca_filesystem.language.lower():
        ds_dict = split_python_dataset(dataset)
        for split_name, ds in ds_dict.items():
            print(split_name, len(ds))
            ds.to_parquet(lca_filesystem.hf_completion_files_dir / f'{split_name}_chunk_{chunk_count :04d}.parquet')
    else:
        dataset.to_parquet(lca_filesystem.hf_completion_files_dir / f'train_chunk_{chunk_count :04d}.parquet')

    chunk_count += 1
    hf_data = list()
    print(len(hf_data))
    return hf_data, chunk_count



@click.command()
@click.option('--language', '-l', default='kotlin')
@click.option('--lca-dir', '-d', default='/mnt/data2/shared-data/lca/')
@click.option('--strategy', '-s', default='dummy')
@click.option('--max-lines', '-m', default=2_000)
def completion_files_to_parquet(language: str, lca_dir: str, strategy: str, max_lines: int):
    lca_filesystem = LCAFilesystem(lca_dir=lca_dir, language=language)
    hf_data = list()

    path_to_tmp_save = lca_filesystem.hf_data_dir / '_tmp'
    if path_to_tmp_save.exists():
        shutil.rmtree(path_to_tmp_save)
        # raise FileExistsError(f'Directory "{path_to_tmp_save}" already exists.')
    path_to_tmp_save.mkdir(exist_ok=True, parents=True)

    chunk_count = 0
    processed_completion_files = [
        ProcessedCompletionFilePath.from_path(path) for path in lca_filesystem.processed_completion_files
    ]
    for processed_completion_file_path in tqdm(processed_completion_files):
        filepath_identifier = asdict(processed_completion_file_path)
        filepath_identifier.pop('path')

        completion_file_path = CompletionFilePath(**filepath_identifier)
        completion_file_path.compose_path(lca_filesystem=lca_filesystem)

        commit_metadata_path = CommitMetadataPath(**filepath_identifier)
        commit_metadata_path.compose_path(lca_filesystem=lca_filesystem)

        with open(processed_completion_file_path.path, 'r') as f:
            _processed_data = json.load(f)
        processed_data = [ProcessedFile.from_json(pf) for pf in _processed_data]
        with open(completion_file_path.path, 'r') as f:
            _completion_data = json.load(f)
        completion_data = [FileModMetadata.from_json(cf) for cf in _completion_data]
        with open(commit_metadata_path.path, 'r') as f:
            _commit_metadata = json.load(f)
        commit_metadata = CommitMetadata(**_commit_metadata)

        if  not hasattr(commit_metadata, 'snapshot_hash'):
            raise ValueError(f'file {str(commit_metadata_path.path)},\n'
                             f'commit metadata does not have `snapshot_hash` field.\n'
                             'Try to run `repo_processing.scripts.add_snapshot_hash_to_commit_metadata`')

        kwargs = dict()

        for processed_file, completion_file in lca_filesystem.completion_pairs_iterator(processed_data, completion_data):
            hf_completion_file = HFCompletionFileLocal.from_file(processed_file, completion_file, commit_metadata,
                                                                 return_dict=True, category_processor_name=strategy,
                                                                 **kwargs)
            if hf_completion_file['total_lines'] <= max_lines:
                hf_data.append(hf_completion_file)

        if len(hf_data) // 100_000 > 0:
            hf_data, chunk_count = write_chunk(hf_data, chunk_count, lca_filesystem,
                                               # path_to_tmp_save=path_to_tmp_save
                                               )


    if len(hf_data) > 0:
        hf_data, chunk_count = write_chunk(hf_data, chunk_count, lca_filesystem,
                                           # path_to_tmp_save=path_to_tmp_save
                                           )

    # for hf_json_path in path_to_tmp_save.rglob('*hf_data_chunk*.json'):
    #     with open(hf_json_path, 'r') as f:
    #         hf_data.extend(json.load(f))
    # shutil.rmtree(path_to_tmp_save)
    #
    # dataset = Dataset.from_list(hf_data, features=completion_file_features_local)
    #
    # if 'python' in lca_filesystem.language.lower():
    #     ds_dict = split_python_dataset(dataset)
    #     for split_name, ds in ds_dict.items():
    #         print(split_name, len(ds))
    #         ds.to_parquet(lca_filesystem.hf_completion_files_dir / f'{split_name}.parquet')
    # else:
    #     dataset.to_parquet(lca_filesystem.hf_completion_files_dir / 'train.parquet')


if __name__ == '__main__':
    completion_files_to_parquet()
