from dataclasses import dataclass
from typing import Any


@dataclass
class BaseEngineOutput:
    output: Any = None

    def get_out_text(self) -> str:
        raise NotImplementedError

    def get_out_extras(self) -> dict:
        raise NotImplementedError


class BaseEngine:
    def generate(self, prompts: list[str], **generation_args) -> list[BaseEngineOutput]:
        raise NotImplementedError

    @staticmethod
    def get_sampling_params(**sampling_params) -> Any:
        raise NotImplementedError
