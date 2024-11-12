from datetime import datetime
from pathlib import Path
from typing import Iterable

from pipeline.data.extension.lca_filesystem.data_classes.parsed_filepathes import CommitMetadataPath, CompletionFilePath, RepoSnapshotPath
from pipeline.data.extension.line_classification.data_classes.processed_file import ProcessedFile
from pipeline.data.extension.repo_processing.data_classes.file_mod_metadata import FileModMetadata


class LCAFilesystem:
    def __init__(self, lca_dir: str | Path, language: str):
        if isinstance(lca_dir, str):
            lca_dir = Path(lca_dir)
        self.lca_dir = lca_dir
        self.language = language
        self.permissive_repos_dir = lca_dir / language / 'permissive_repos'
        self.completion_files_dir = lca_dir / language / 'completion_files'
        self.processed_files_dir = lca_dir / language / 'processed_files'
        self.repo_snapshots_dir = lca_dir / language / 'repo_snapshots'
        self.hf_data_dir = lca_dir / language / 'hf_data'
        self.metainfo_dir = lca_dir / language / 'metainfo'
        self.logs_dir = lca_dir / language / '_logs'

        self._resolve_dirs()

    def _resolve_dirs(self):
        self.permissive_repos_dir.mkdir(parents=True, exist_ok=True)
        self.completion_files_dir.mkdir(parents=True, exist_ok=True)
        self.processed_files_dir.mkdir(parents=True, exist_ok=True)
        self.repo_snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.hf_data_dir.mkdir(parents=True, exist_ok=True)
        self.metainfo_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def repo_snapshots(self) -> list[Path]:
        return list(self.repo_snapshots_dir.rglob('*.json*'))

    @property
    def processed_completion_files(self) -> list[Path]:
        return list(self.processed_files_dir.rglob('*/processed_*.json*'))

    @property
    def commit_info_files(self) -> list[Path]:
        return list(self.completion_files_dir.rglob('*/info_*.json'))

    @property
    def completion_files(self) -> list[Path]:
        return list(self.completion_files_dir.rglob('*/modified_files_*.json'))

    @property
    def permissive_repos(self) -> list[Path]:
        return [Path(fp) for fp in self.permissive_repos_dir.iterdir() if Path(fp).is_dir()]

    def _get_commits_dict(self) -> dict[str, list[str]]:
        """
        This method analyzes existing completion files and returns a dictionary of commit hashes for each repository.
        :return: A dictionary where the keys are repository paths as strings and the values are lists of commit file hashes.
        :rtype: dict[str, list[str]]
        """
        commits_dict = dict()
        for repo_path in self.completion_files_dir.iterdir():
            repo_path = Path(repo_path)
            info_commit_files = [CommitMetadataPath.from_path(fp) for fp in repo_path.rglob('info_*.json')]
            mod_commit_files = [CompletionFilePath.from_path(fp) for fp in repo_path.rglob('modified_files_*.json')]

            if (sorted([info.completion_file_hash for info in info_commit_files]) !=
                    sorted([mf.completion_file_hash for mf in mod_commit_files])):
                raise ValueError(f'info and modified_files jsons are not in correspondence in {str(repo_path)}')
            commits_dict[str(repo_path.name)] = [info.completion_file_hash for info in info_commit_files]
        return commits_dict

    def _get_commits(self) -> list[tuple[str, str]]:
        """
        Commits Dict flattener
        :return: A list of tuples where each tuple contains the repository name and the respective commit hash.
        """
        commits = list()
        commits_dict = self.commits_dict
        for repo_name, hashes in commits_dict.items():
            for commit_hash in hashes:
                commits.append((repo_name, commit_hash))
        return commits

    @property
    def commits(self) -> list[tuple[str, str]]:
        return self._get_commits()

    @property
    def commits_dict(self) -> dict[str, list[str]]:
        return self._get_commits_dict()

    def logs_dir_for_script(self, script_path: Path) -> Path:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        script_name = script_path.stem
        run_logs_dir = self.logs_dir / script_name / timestamp_str
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        return run_logs_dir

    def completion_pairs_iterator(self,
                                  processed_data: list[ProcessedFile],
                                  completion_data: list[FileModMetadata]
                                  ) -> Iterable[tuple[ProcessedFile, FileModMetadata]]:
        if len(processed_data) != len(completion_data):
            print('data have different len')
        for processed_file in processed_data:
            for completion_file in completion_data:
                if processed_file.filename == completion_file.path:
                    yield processed_file, completion_file

    @property
    def hf_repo_snapshots_dir(self) -> Path:
        rs_dir = self.hf_data_dir / 'repo_snapshots'
        rs_dir.mkdir(exist_ok=True)
        return rs_dir

    def hf_repo_snapshot_num_chars_dirs(self, num_char_names: list[str]):
        for name in num_char_names:
            (self.hf_repo_snapshots_dir / name).mkdir(exist_ok=True, parents=True)
        return [Path(nc_dir) for nc_dir in self.hf_repo_snapshots_dir.iterdir() if Path(nc_dir).is_dir()]

    @property
    def completion_hash_to_repo_snapshot_path(self) -> dict[str, Path]:
        repo_paths = [RepoSnapshotPath.from_path(p) for p in self.repo_snapshots]
        if len([repo_path.completion_file_hash for repo_path in repo_paths]) != len({repo_path.completion_file_hash for repo_path in repo_paths}):
            raise ValueError('Completion Hash is not enough to uniquely encode')
        completion_hash_to_repo_snapshot_path = {
            repo_path.completion_file_hash: repo_path.path for repo_path in repo_paths
        }
        return completion_hash_to_repo_snapshot_path

    @property
    def hf_completion_files_data_files(self) -> list[str]:
        # TODO: Implement choosing split names
        return [str(_parquet_path) for _parquet_path in self.hf_completion_files_dir.rglob('train*.parquet')]

    def hf_repo_snapshot_path(self, repo_name: str, num_chars_name: str) -> Path:
        return self.hf_repo_snapshots_dir / num_chars_name / (repo_name +'.parquet')

    @property
    def hf_completion_files_dir(self) -> Path:
        hf_cf_dir = self.hf_data_dir / 'completion_files_local'
        hf_cf_dir.mkdir(parents=True, exist_ok=True)
        return hf_cf_dir
