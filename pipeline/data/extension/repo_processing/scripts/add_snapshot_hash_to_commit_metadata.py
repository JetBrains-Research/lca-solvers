import json

import click
from tqdm import tqdm

from pipeline.data.extension.lca_filesystem.data_classes.parsed_filepathes import RepoSnapshotPath, CommitMetadataPath
from pipeline.data.extension.lca_filesystem.lca_filesystem import LCAFilesystem
from pipeline.data.extension.lca_filesystem.num_chars_utils import get_num_char_name
from pipeline.data.extension.repo_processing.data_classes.commit_metadata import CommitMetadata


@click.command()
@click.option('--language', '-l', default='kotlin')
@click.option('--lca-dir', '-d', default='/mnt/data2/shared-data/lca/')
def add_snapshot_hash_to_commit_metadata(language: str, lca_dir: str):
    lca_filesystem = LCAFilesystem(lca_dir, language)
    snapshot_file_paths = [RepoSnapshotPath.from_path(repopath) for repopath in lca_filesystem.repo_snapshots]
    for repopath in tqdm(snapshot_file_paths):
        metadata_parsed_path = CommitMetadataPath(repo=repopath.repo, completion_file_hash=repopath.completion_file_hash)
        metadata_path = lca_filesystem.completion_files_dir / metadata_parsed_path.relative_path
        if metadata_path.is_file():
            with open(metadata_path, 'r') as f:
                commit_metadata = CommitMetadata.from_json(json.load(f))
            if commit_metadata.snapshot_hash is None:
                commit_metadata.snapshot_hash = repopath.snapshot_hash
            elif commit_metadata.snapshot_hash != repopath.snapshot_hash:
                raise ValueError(f'For {metadata_path} snapshot_hash exists and different from {repopath}')
            with open(repopath.path, 'r') as f:
                _repo_snapshot_data = json.load(f)
            commit_metadata.repo_num_chars_relevant = _repo_snapshot_data['num_chars_relevant']
            commit_metadata.repo_num_chars_total = _repo_snapshot_data['num_chars_total']
            commit_metadata.repo_num_chars_name = get_num_char_name(commit_metadata.repo_num_chars_relevant)
            commit_metadata.info_to_disk(lca_filesystem=lca_filesystem, dir_name=metadata_path.parent.name)


if __name__ == '__main__':
    add_snapshot_hash_to_commit_metadata()
