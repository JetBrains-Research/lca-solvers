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
class OldDatapoint:
    repo: str
    commit_hash: str
    completion_file: CompletionFile
    completion_lines: CompletionLines
    repo_snapshot: RepoSnapshot
    completion_lines_raw: CompletionLines | None = None

    def recalculate_random_category(self) -> None:
        pass


def Datapoint(repo: str,
              commit_hash: str,
              filename: str,
              content: str,
              **_kwargs,
              ) -> OldDatapoint:
    return OldDatapoint(
        repo=repo,
        commit_hash=commit_hash,
        completion_file=CompletionFile(
            filename=filename,
            content=content,
        ),
        completion_lines=CompletionLines(
            commited=[],
            common=[],
            infile=[],
            inproject=[],
            non_informative=[],
            random=[],
        ),
        repo_snapshot=RepoSnapshot(
            filename=[],
            content=[],
        ),
    )


class BatchDatapoint(TypedDict):
    repo: list[str]
    commit_hash: list[str]
    completion_file: list[CompletionFile]
    completion_lines: list[CompletionLines]
    repo_snapshot: list[RepoSnapshot]
    completion_lines_raw: list[CompletionLines]
