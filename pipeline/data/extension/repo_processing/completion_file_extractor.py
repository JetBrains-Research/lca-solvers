from pathlib import Path
from typing import Iterator

from pydriller import Repository, Commit, ModifiedFile, ModificationType
import os

from pipeline.data.extension.lca_filesystem.lca_filesystem import LCAFilesystem
from pipeline.data.extension.repo_processing.data_classes.commit_metadata import CommitMetadata
from pipeline.data.extension.repo_processing.data_classes.file_mod_metadata import FileModMetadata
# from repo_processing.repo_storage import RepoStorage


class CommitChecker:
    def __init__(self, commit: Commit):
        self.commit = commit

    def is_file_added(self) -> bool:
        return any(file_mod.change_type == ModificationType.ADD for file_mod in self.commit.modified_files)

    def check_extensions(self, extensions: list[str] | None) -> bool:
        if extensions is None:
            return True
        _check_file_extension = lambda filepath: any(filepath.endswith(ext) for ext in extensions)
        return any(_check_file_extension(file_mod.new_path) for file_mod in self.commit.modified_files if file_mod.new_path)


class CompletionFileExtractor:
    def __init__(self,
                 repo_path: str | Path,
                 lca_filesystem: LCAFilesystem,
                 min_code_lines: int = -1,
                 max_code_lines: int = 1_000,
                 allowed_extensions: list[str] | None = None
                 ):
        self.repo_path = Path(repo_path)
        self.lca_filesystem = lca_filesystem
        self.min_code_lines = min_code_lines
        self.max_code_lines = max_code_lines
        self.repo_name = repo_path.name
        self.repository = Repository(str(self.repo_path))
        self._allowed_extensions = allowed_extensions

    def extract(self,) -> Iterator[CommitMetadata | str]:
        for commit in self.repository.traverse_commits():
            commit_metadata = self._process_commit(commit)
            if commit_metadata is None:
                yield commit.hash
            else:
                commit_metadata.mods = [
                    mod_file for mod_file in commit_metadata.mods
                    if self._check_mod_file(mod_file)
                ]
                if len(commit_metadata.mods) > 0:
                    yield commit_metadata

    def _check_commit(self, commit: Commit) -> bool:
        commit_processor = CommitChecker(commit)
        return commit_processor.is_file_added() and commit_processor.check_extensions(self._allowed_extensions)

    def _check_mod_file(self, mod_file_metadata: FileModMetadata,) -> bool:
        # Check extension
        if mod_file_metadata.path is None:
            return False
        if self._allowed_extensions:
            if not any(mod_file_metadata.path.endswith(ext) for ext in self._allowed_extensions):
                return False
        # Check modification type with corner case of moving file
        if mod_file_metadata.mod_type != 'ADD':
            return False
        # Filter out files with None content
        if mod_file_metadata.content is None:
            return False

        return True

    def _process_commit(self, commit: Commit) -> None | CommitMetadata:
        if self._check_commit(commit):
            commit_metadata = CommitMetadata.from_commit(commit, self.repo_name)
            return commit_metadata
        return None

    @property
    def lca_dir(self) -> str:
        return str(self.lca_filesystem.lca_dir)


if __name__ == "__main__":
    # repo_storage = RepoStorage(
    #     lca_dir = '/mnt/data2/shared-data/lca/',
    #     language = 'kotlin'
    # )
    # repo_path = repo_storage[100]
    lca_dir = '/Users/Evgeniy.Glukhov/Datasets/lca'
    language = 'kotlin'
    lca_filesystem = LCAFilesystem(lca_dir=lca_dir, language=language)
    repo_path = lca_filesystem.permissive_repos[1]
    extractor = CompletionFileExtractor(
        repo_path=repo_path,
        lca_filesystem=lca_filesystem,
        allowed_extensions=['.kt'],
        # min_code_lines=50,
        # max_code_lines=200,
    )
    for cm in extractor.extract():
        print(len(cm.mods))
        break
