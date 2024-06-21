from dataclasses import dataclass
from typing import Optional

from vllm import LLM, RequestOutput, SamplingParams

from lca_eval_harness.model_inference.base_engine import BaseEngine, BaseEngineOutput


@dataclass
class VllmEngineOutput(BaseEngineOutput):
    output: Optional[RequestOutput] = None

    def get_out_text(self) -> str:
        return self.output.outputs[0].text

    def get_out_extras(self) -> dict:
        return {'engine': 'vllm', 'cumulative_logprob': self.output.outputs[0].cumulative_logprob}


class VllmEngine(BaseEngine):
    def __init__(self, hf_model_path: str, **llm_args):
        self.hf_model_path = hf_model_path
        self.llm = self._get_vllm(**llm_args)

    def _get_vllm(self, **llm_args) -> LLM:
        return LLM(model=self.hf_model_path, **llm_args)

    def generate(self, prompts: list[str], **generation_args) -> list[VllmEngineOutput]:
        outputs = [VllmEngineOutput(output=output) for output in self.llm.generate(prompts=prompts, **generation_args)]
        return outputs

    @staticmethod
    def get_sampling_params(**sampling_params) -> SamplingParams:
        """
        Method to get sampling parameters for generation out of a dictionary.

        :param sampling_params: see the official implementation
        https://github.com/vllm-project/vllm/blob/e2b85cf86a522e734a38b1d0314cfe9625003ef9/vllm/sampling_params.py#L31
        :return: instance of vllm.SamplingParams that is used for generation
        """
        return SamplingParams(**sampling_params)


if __name__ == '__main__':
    inference_engine = VllmEngine(hf_model_path='deepseek-ai/deepseek-coder-1.3b-base')
    sampling_params = inference_engine.get_sampling_params(
        temperature=0.0,
        min_tokens=15,
        max_tokens=150,
    )
    outputs = inference_engine.generate(prompts=['#python\ndef hello', 'class Hello'], sampling_params=sampling_params)
    for output in outputs:
        print(output.output.prompt + output.output.outputs[0].text)
        print('\n', '-'*100, '\n\n')
