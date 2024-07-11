from dataclasses import dataclass
from typing import TypedDict


class CompletionFile(TypedDict):
    filename: str
    content: str


class CompletionLines(TypedDict, total=False):
    commited: list[int]
    common: list[int]
    infile: list[int]
    inproject: list[int]
    non_informative: list[int]
    random: list[int]
    other: list[int]


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
    completion_lines_raw: CompletionLines | None = None


class BatchDatapoint(TypedDict):
    repo: list[str]
    commit_hash: list[str]
    completion_file: list[CompletionFile]
    completion_lines: list[CompletionLines]
    repo_snapshot: list[RepoSnapshot]
    completion_lines_raw: list[CompletionLines]
