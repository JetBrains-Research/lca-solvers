import json
from dataclasses import dataclass, asdict
from enum import nonmember

from pipeline.data.extension.lca_filesystem.lca_filesystem import LCAFilesystem
from pipeline.data.extension.repo_processing.data_classes.file_mod_metadata import FileModMetadata
from pydriller import Commit


@dataclass
class CommitMetadata:
    repo: str
    hash: str
    author_date: str
    author_timezone: int
    committer_date: str
    committer_timezone: int
    mods: list[FileModMetadata] | None = None
    snapshot_hash: str | None = None
    repo_num_chars_relevant: int = -1
    repo_num_chars_total: int = -1
    repo_num_chars_name: str | None = None

    def __post_init__(self):
        if self.mods is not None:
            self.identify_renaming()

    def identify_renaming(self):
        rename_count = 0
        for i, mod in enumerate(self.mods):
            if mod.mod_type == 'ADD' and mod.content is not None:
                if mod.content.strip() in self.deleted_files:
                    self.mods[i].mod_type = 'RENAME'
                    rename_count += 1
        if rename_count > 0:
            print('Renamed Files:', rename_count)


    @classmethod
    def from_commit(cls, commit: Commit, repo_name: str) -> 'CommitMetadata':
        commit_data = {
            "repo": repo_name,
            "hash": commit.hash,
            "author_date": commit.author_date.strftime("%d.%m.%Y %H:%M:%S"),
            "author_timezone": commit.author_timezone,
            "committer_date": commit.committer_date.strftime("%d.%m.%Y %H:%M:%S"),
            "committer_timezone": commit.committer_timezone,
            "mods": [],
        }
        for mod in commit.modified_files:
            try:
                commit_data["mods"].append(FileModMetadata.from_modified_file(mod))
            except ValueError as e:
                print(e, mod.old_path, mod.new_path, mod.change_type)
                pass
        return cls(**commit_data)

    @property
    def deleted_files(self) -> list[str]:
        return [mf.content.strip() for mf in self.mods if (mf.mod_type == 'DELETE' and mf.content is not None)]

    @classmethod
    def from_json(cls, data: dict) -> 'CommitMetadata':
        return cls(**data)

    def info_to_disk(self, lca_filesystem: LCAFilesystem, dir_name: str) -> str:
        metadata = asdict(self)
        metadata['mods'] = None
        metadata_filename = f'info_{self.hash}.json'
        saving_filepath = lca_filesystem.completion_files_dir / dir_name / metadata_filename
        with open(saving_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        return str(saving_filepath)

    def modified_files_to_disk(self, lca_filesystem: LCAFilesystem, dir_name: str) -> str:
        modified_files = [asdict(mod_file) for mod_file in self.mods]
        modified_files_filename = f'modified_files_{self.hash}.json'
        modified_files_filepath = lca_filesystem.completion_files_dir / dir_name / modified_files_filename
        with open(modified_files_filepath, 'w') as f:
            json.dump(modified_files, f, indent=4)
        return str(modified_files_filepath)

    def to_disk(self, lca_filesystem: LCAFilesystem, dir_name: str) -> (str, str):
        info_path = self.info_to_disk(lca_filesystem, dir_name)
        modified_files_path = self.modified_files_to_disk(lca_filesystem, dir_name)
        return info_path, modified_files_path
