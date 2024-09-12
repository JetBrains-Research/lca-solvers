from composers.data.datapoint import CompletionLines

from typing import TypedDict


class ComposedDatapoint(TypedDict):
    pre_context_prompt: str
    composed_context: str
    composed_completion: str
    completion_lines: CompletionLines


class BatchComposedDatapoint(TypedDict):
    pre_context_prompt: list[str]
    composed_context: list[str]
    composed_completion: list[str]
    completion_lines: list[CompletionLines]
