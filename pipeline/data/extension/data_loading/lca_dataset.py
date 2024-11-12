import json
from typing import Iterable

from datasets import Dataset, load_dataset

from pipeline.data.extension.hf_processing.data_classes.hf_completion_file_local import HFCompletionFileLocal
from pipeline.data.extension.hf_processing.hf_features.completion_file_features import completion_file_features_local
from pipeline.data.extension.hf_processing.hf_features.repo_snapshot_features import snapshot_features
from pipeline.data.extension.lca_filesystem import LCAFilesystem
from pipeline.data.extension.lca_filesystem.data_classes.parsed_filepathes import RepoSnapshotPath
from pipeline.data.extension.repo_processing.data_classes.repo_snapshot import FileSnapshot


class LCADataset:
    IMPLEMENTED_REPO_SOURCES = ['none', 'json', 'parquet']

    def __init__(self, lca_dir: str, language: str, repo_snapshots_source: str = 'none'):
        self.lca_dir = lca_dir
        self.language = language
        self.lca_filesystem = LCAFilesystem(lca_dir, language)
        if repo_snapshots_source not in self.IMPLEMENTED_REPO_SOURCES:
            raise ValueError(
                f'`repo_snapshots_source` argument must be one of {self.IMPLEMENTED_REPO_SOURCES}, '
                f'got {repo_snapshots_source}')
        self._repo_snapshots_source = repo_snapshots_source

        self._completion_files: Dataset = self._load_completion_files()

    def _load_completion_files(self) -> Dataset:
        return load_dataset('parquet', data_files=self.lca_filesystem.hf_completion_files_data_files, split='train',
                            features=completion_file_features_local)

    def __len__(self):
        return len(self._completion_files)

    def __iter__(self) -> Iterable[tuple[HFCompletionFileLocal, list[FileSnapshot]]]:
        for completion_file in self._completion_files:
            hf_completion_file = HFCompletionFileLocal.from_hf_dataset(completion_file)
            repo_files: list[FileSnapshot] = list()
            if self._repo_snapshots_source == 'none':
                pass  # No repo loading when the source is 'none'
            elif self._repo_snapshots_source == 'json':
                repo_snapshot_path = RepoSnapshotPath(repo=hf_completion_file.repo,
                                                      year=hf_completion_file.year,
                                                      snapshot_hash=hf_completion_file.snapshot_hash,
                                                      completion_file_hash=hf_completion_file.commit_hash)
                repo_snapshot_path.compose_path(self.lca_filesystem)
                with open(repo_snapshot_path.path, 'r') as f:
                    _repo = json.load(f)
                repo_files = [FileSnapshot(**repo_file) for repo_file in _repo['files']]

            elif self._repo_snapshots_source == 'parquet':
                if hf_completion_file.repo_num_chars_name is None:
                    pass
                else:
                    repo_snapshot_path = self.lca_filesystem.hf_repo_snapshot_path(hf_completion_file.repo,
                                                                                   hf_completion_file.repo_num_chars_name)
                    repo_snapshot_path = str(repo_snapshot_path)
                    print(repo_snapshot_path)
                    repo_snapshots = load_dataset('parquet', data_files=[repo_snapshot_path], split='train',
                                                 features=snapshot_features)

                    for _file_data in repo_snapshots:
                        if _file_data['completion_file_commit_hash'] == hf_completion_file.commit_hash:
                            repo_files.append(FileSnapshot(filename=_file_data['filename'], content=_file_data['content']))

            else:
                raise NotImplementedError

            yield hf_completion_file, repo_files

    def get_completion_files(self) -> Dataset:
        return self._completion_files

    def __getitem__(self, item):
        return self._completion_files[item]

if __name__ == '__main__':
    ds = LCADataset(lca_dir='/mnt/data/shared-data/lca/plcc_data_dir/', language='python', repo_snapshots_source='none')
    print(ds._completion_files)
    for cf, rs in ds:
        print(cf)
        print(rs)
        break
