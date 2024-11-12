from dataclasses import dataclass
from typing import TypedDict


class CompletionFile(TypedDict):
    filename: str
    content: str


class CompletionLines(TypedDict, total=False):
    InCommit: list[int]
    InFile: list[int]
    InProject: list[int]
    NonInformative: list[int]
    Other: list[int]
    OtherAPI: list[int]
    TODO: list[int]


class RepoSnapshot(TypedDict):
    filename: list[str]
    content: list[str]


@dataclass
class Datapoint:
    repo: str
    commit_hash: str
    completion_file: CompletionFile
    completion_lines: CompletionLines
    repo_snapshot: RepoSnapshot

    def recalculate_random_category(self) -> None:
        pass


class BatchDatapoint(TypedDict):
    repo: list[str]
    commit_hash: list[str]
    completion_file: list[CompletionFile]
    completion_lines: list[CompletionLines]
    repo_snapshot: list[RepoSnapshot]
