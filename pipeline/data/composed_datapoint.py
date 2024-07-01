from typing import TypedDict


class ComposedDatapoint(TypedDict):
    pre_context_prompt: str
    composed_context: str
    composed_completion: str

class BatchedComposedDatapoint(TypedDict):
    pre_context_prompt: list[str]
    composed_context: list[str]
    composed_completion: list[str]
