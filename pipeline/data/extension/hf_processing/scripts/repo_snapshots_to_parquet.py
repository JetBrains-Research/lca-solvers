import json
from pathlib import Path

import click
from datasets import Dataset

from pipeline.data.extension.hf_processing.hf_features.repo_snapshot_features import snapshot_features
from pipeline.data.extension.lca_filesystem.num_chars_utils import num_char_names, get_num_char_name
from pipeline.data.extension.lca_filesystem import LCAFilesystem
from pipeline.data.extension.lca_filesystem.data_classes.parsed_filepathes import RepoSnapshotPath


@click.command()
@click.option('--language', '-l', default='kotlin')
@click.option('--lca-dir', '-d', default='/mnt/data2/shared-data/lca/')
def repo_snapshots_to_parquet(language: str, lca_dir: str):
    lca_filesystem = LCAFilesystem(lca_dir=lca_dir, language=language)
    run_logs_dir = lca_filesystem.logs_dir_for_script(Path(__file__))

    num_chars_dirs = lca_filesystem.hf_repo_snapshot_num_chars_dirs(num_char_names=num_char_names)

    errors = list()
    repo_snapshot_from_json_to_num_chars = dict()

    for repo_snapshots_path in lca_filesystem.repo_snapshots_dir.iterdir():
        if not repo_snapshots_path.is_dir():
            continue
        repo_snap_dict = {name: list() for name in num_char_names}

        for rs_path in repo_snapshots_path.rglob('*.json'):
            relative_path = RepoSnapshotPath.from_path(rs_path).relative_path
            with open(rs_path, 'r') as f:
                data = json.load(f)
            data_name = get_num_char_name(data['num_chars_relevant'])
            if data_name is not None:
                files_list = data.pop('files')
                for file_data in files_list:
                    file_data['completion_file_commit_hash'] = data['completion_file_commit_hash']
                    file_data['repo_snapshot_commit_hash'] = data['commit_hash']
                repo_snap_dict[data_name].extend(files_list)
                repo_snapshot_from_json_to_num_chars[relative_path] = data_name
            if data_name is None:
                with open(run_logs_dir / 'non_num_chars_snapshots.log', 'a') as f:
                    f.write(relative_path + '\n')

        try:
            for name, ds_list in repo_snap_dict.items():
                if len(ds_list) > 0:
                    ds = Dataset.from_list(ds_list, features=snapshot_features)
                    ds.to_parquet(lca_filesystem.hf_repo_snapshot_path(repo_snapshots_path.name, name))
        except Exception as e:
            print('.')
            errors.append([repo_snapshots_path, e])

    with open(run_logs_dir / 'errors.log', 'w') as f:
        for err in errors:
            f.write(f'{str(err[0])};{str(err[1])};' + '\n')

    with open(lca_filesystem.metainfo_dir / 'repo_snapshot_from_json_to_num_chars.json', 'w') as f:
        json.dump(repo_snapshot_from_json_to_num_chars, f, indent=4)

if __name__ == '__main__':
    repo_snapshots_to_parquet()
