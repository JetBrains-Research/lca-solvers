from pipeline.data.datapoint import CompletionLines

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


class ComposedBlockDatapoint(TypedDict):
    context_blocks: list[str]
    completion_block: str
    completion_lines: CompletionLines


class BatchComposedBlockDatapoint(TypedDict):
    context_blocks: list[list[str]]
    completion_block: list[str]
    completion_lines: list[CompletionLines]
