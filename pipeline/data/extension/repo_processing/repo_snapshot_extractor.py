import logging
import time
from pathlib import Path

import chardet
import git

from git.objects.blob import Blob

from joblib import Parallel, delayed

from pipeline.data.extension.lca_filesystem.lca_filesystem import LCAFilesystem
from pipeline.data.extension.repo_processing._binary_extensions import binary_extensions
from pipeline.data.extension.repo_processing.data_classes.repo_snapshot import FileSnapshot, RepoSnapshot


class RepoSnapshotExtractor:
    def __init__(self,
                 # lca_dir: str, language: str,
                 lca_filesystem: LCAFilesystem,
                 repo_name: str, commit_hash: str):
        self.lca_filesystem = lca_filesystem
        # self.lca_dir = lca_dir
        # self.language = language
        self.repo_name = repo_name
        self.commit_hash = commit_hash
        self.repo_path = self._get_repo_path()
        # self.repo_snapshots_dir_name = 'repo_snapshots'
        self.all_repo_snapshots_dir = self.lca_filesystem.repo_snapshots_dir #os.path.join(self.lca_dir, self.language, self.repo_snapshots_dir_name)
        self.repo_snapshot_dir = self.all_repo_snapshots_dir / self.repo_name

    def extract(self,
                num_workers: int = -1,
                relevant_extensions: list[str] | None = None,
                save_to_disk: bool = True, ) -> RepoSnapshot | None:
        repo_snapshot = self.get_repo_snapshot_before_commit(num_workers)
        if repo_snapshot is None:
            return repo_snapshot
        if relevant_extensions is not None:
            repo_snapshot.get_num_chars_relevant(relevant_extensions)
        if save_to_disk:
            filename_on_disk = f'snapshot_{repo_snapshot.commit_hash}_completion_{repo_snapshot.completion_file_commit_hash}.json'
            year_dir = self.repo_snapshot_dir / str(repo_snapshot.year)
            year_dir.mkdir(parents=True, exist_ok=True)
            # os.makedirs(year_dir, exist_ok=True)
            filepath_on_disk = year_dir / filename_on_disk
            repo_snapshot.to_disk(filepath_on_disk)
        return repo_snapshot


    def get_repo_snapshot_before_commit(self, num_workers: int = -1) -> RepoSnapshot | None:
        repo = git.Repo(self.repo_path)
        commit = repo.commit(self.commit_hash)
        year = commit.committed_datetime.year

        if commit.parents:
            parent_commit = commit.parents[0]
            parent_hash = parent_commit.hexsha
        else:
            return None
        # repo.remotes.origin.fetch()
        # repo.git.reset('--hard')  # Discard local changes
        # repo.git.clean('-fd')  # Remove untracked files and directories
        # repo.git.checkout(parent_hash,) # force=True)

        files: list[FileSnapshot] = list()

        blobs = [blob for blob in parent_commit.tree.traverse() if blob.type == "blob"]

        if num_workers <= 1:
            for blob in blobs:
                file_snapshot = self._process_blob(blob)
                files.append(file_snapshot)
        elif num_workers > 1:
            with Parallel(num_workers) as pool:
                files = pool(delayed(self._process_blob)(blob) for blob in blobs)

        files = [file_snapshot for file_snapshot in files if file_snapshot is not None]

        repo_snapshot = RepoSnapshot(
            files=files,
            year=year,
            repo=self.repo_name,
            commit_hash=parent_hash,
            completion_file_commit_hash=self.commit_hash,
        )

        return repo_snapshot

    def _process_blob(self, blob: Blob) -> FileSnapshot:
        file_path = str(blob.path)
        try:
            if self._check_binary_extension(file_path):
                return FileSnapshot(filename=file_path, content='')
            blob_stream = blob.data_stream
            blob_data = blob_stream.read()
            content = self._decode_text(blob_data, file_path)
            file_snapshot = FileSnapshot(filename=file_path, content=str(content))
        except Exception as e:
            file_snapshot = FileSnapshot(filename=file_path, content='')
            raise e
            logging.warning(
                f"Could not read file '{file_path}'. Replaced with empty string. Error: {e}")
        return file_snapshot

    def _check_binary_extension(self, filepath: str, binary_extensions: set[str] = binary_extensions) -> bool:
        ext = Path(filepath).suffix
        return ext in binary_extensions

    def _decode_text(self, blob_data: bytes, file_path: str) -> str:
        encoding = 'utf-8'
        replacement_char = '\uFFFD'
        content = blob_data.decode(encoding, errors='replace')
        error_char_count = content.count(replacement_char)
        total_char_count = len(content)
        if total_char_count == 0:
            return ''
        if error_char_count / total_char_count > 0.1:
            result = chardet.detect(blob_data)
            encoding = result['encoding']
            if encoding is None:
                return ''
            try:
                content = blob_data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                print(file_path, end='')
                print(result)
                content = ''

        return content


    def _get_repo_path(self) -> Path:
        return self.lca_filesystem.permissive_repos_dir / self.repo_name


if __name__ == "__main__":
    # cf_storage = CompletionFilesStorage(
    #     lca_dir='/mnt/data2/shared-data/lca/',
    #     language='kotlin'
    # )
    lca_dir = '/mnt/data2/shared-data/lca/'
    language = 'kotlin'
    lca_filesystem = LCAFilesystem(lca_dir=lca_dir, language=language)
    for repo_name, commit_hash in lca_filesystem.commits[5000:]:
        rs_extractor = RepoSnapshotExtractor(
            lca_filesystem=lca_filesystem,
            repo_name=repo_name,
            commit_hash=commit_hash
        )

        start = time.time()
        repo_snapshot = rs_extractor.extract(relevant_extensions=['kt',])
        finish = time.time()
        print(f'Not parallel time: {finish - start}')
        # num_workers = 2
        # start = time.time()
        # repo_snapshot_parallel = rs_extractor.extract(num_workers)
        # finish = time.time()
        # print(f'Parallel with {num_workers} workers, time: {finish - start}')

        # print(repo_snapshot == repo_snapshot_parallel)

        print(repo_snapshot.repo, repo_snapshot.year, len(repo_snapshot))
        for file in repo_snapshot:
            if 'git' in file.filename:
                print(file.filename, len(file.content.split('\n')))
                print('-' * 100)
        break
