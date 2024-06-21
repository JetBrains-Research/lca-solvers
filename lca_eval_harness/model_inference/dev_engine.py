from dataclasses import dataclass
from typing import Optional

from lca_eval_harness.model_inference.base_engine import BaseEngine, BaseEngineOutput


@dataclass
class DevEngineOutput(BaseEngineOutput):
    output: Optional[str] = None

    def get_out_text(self) -> str:
        return self.output

    def get_out_extras(self) -> dict:
        return {'engine': 'dev'}


class DevEngine(BaseEngine):
    def generate(self, prompts: list[str], **generation_args) -> list[DevEngineOutput]:
        return [DevEngineOutput(output='dev generation') for _ in prompts]

    @staticmethod
    def get_sampling_params(**sampling_params) -> None:
        return None
