from dataclasses import dataclass
from pathlib import Path

# from ..lca_filesystem import LCAFilesystem


@dataclass
class RepoSnapshotPath:
    repo: str
    year: int
    snapshot_hash: str
    completion_file_hash: str
    path: Path | None = None

    @property
    def filename(self) -> str:
        return f'snapshot_{self.snapshot_hash}_completion_{self.completion_file_hash}.json'

    @property
    def relative_path(self) -> str:
        return f'{self.repo}/{str(self.year)}/{self.filename}'

    @classmethod
    def from_path(cls, path: Path) -> 'RepoSnapshotPath':
        year = int(path.parent.name)
        repo = path.parent.parent.name
        snapshot_hash = path.stem.split('_')[1]
        completion_file_hash = path.stem.split('_')[-1]
        return cls(repo, year, snapshot_hash, completion_file_hash, path=path)

    def compose_path(self, lca_filesystem: 'LCAFilesystem') -> None:
        self.path = lca_filesystem.repo_snapshots_dir / self.relative_path

@dataclass
class CompletionFilePath:
    repo: str
    completion_file_hash: str
    path: Path | None = None

    @property
    def filename(self) -> str:
        return f'modified_files_{self.completion_file_hash}.json'

    @property
    def relative_path(self) -> str:
        return f'{self.repo}/{self.filename}'

    @classmethod
    def from_path(cls, path: Path,) -> 'CompletionFilePath':
        repo = path.parent.name
        completion_file_hash = path.stem.split('_')[-1]
        return cls(repo, completion_file_hash, path=path)

    def compose_path(self, lca_filesystem: 'LCAFilesystem') -> None:
        self.path = lca_filesystem.completion_files_dir / self.relative_path

class CommitMetadataPath(CompletionFilePath):
    @property
    def filename(self) -> str:
        return f'info_{self.completion_file_hash}.json'

    def compose_path(self, lca_filesystem: 'LCAFilesystem') -> None:
        self.path = lca_filesystem.completion_files_dir / self.relative_path

class ProcessedCompletionFilePath(CompletionFilePath):
    @property
    def filename(self) -> str:
        return f'processed_{self.completion_file_hash}.json'

    def compose_path(self, lca_filesystem: 'LCAFilesystem') -> None:
        self.path = lca_filesystem.processed_files_dir / self.relative_path
