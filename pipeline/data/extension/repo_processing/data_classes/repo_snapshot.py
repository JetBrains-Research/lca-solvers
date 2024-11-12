import json
from dataclasses import dataclass, asdict


@dataclass
class FileSnapshot:
    filename: str
    content: str

    def __len__(self):
        return len(self.content)

@dataclass
class RepoSnapshot:
    files: list[FileSnapshot]
    year: int  # Corresponding Completion File Commit Year
    repo: str
    commit_hash: str
    completion_file_commit_hash: str
    relevant_extensions: list[str] | None = None
    num_chars_relevant: int | None = None
    num_chars_total: int = -1

    def __post_init__(self):
        if self.num_chars_total < 0:
            num_chars = 0
            for file in self.files:
                num_chars += len(file)
            self.num_chars_total = num_chars

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return self.files[item]

    def __iter__(self):
        return iter(self.files)

    def to_disk(self, path: str):
        data = asdict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def get_num_chars_relevant(self, relevant_extensions: list[str]):
        self.relevant_extensions = relevant_extensions
        num_chars = 0
        for file_snapshot in self.files:
            if any(file_snapshot.filename.endswith(ext) for ext in relevant_extensions):
                num_chars += len(file_snapshot)
        self.num_chars_relevant = num_chars
