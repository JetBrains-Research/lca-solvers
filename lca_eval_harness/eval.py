import os
import random
from typing import Any

import jsonlines
from vllm import RequestOutput, SamplingParams

from lca_eval_harness.dataset_loaders.hf_data_loader import HFDataLoader
from lca_eval_harness.model_inference.vllm_engine import VllmEngine
from lca_eval_harness.prompters.prompter_base import PrompterBase


class Evaluator:
    # TODO: implement prefix caching https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html
    def __init__(self,
                 inference_engine: VllmEngine,
                 data_loader: HFDataLoader,
                 sampling_params: dict[str, Any],
                 prompter: PrompterBase,
                 artifacts_path: str = 'data/eval_results',
                 ):
        self.inference_engine: VllmEngine = inference_engine
        self.data_loader: HFDataLoader = data_loader
        self.sampling_params: SamplingParams = self.inference_engine.get_sampling_params(**sampling_params)
        self.prompter: PrompterBase = prompter
        self.prompts: list[str] | None = None
        self.targets: list[str] | None = None
        self.outputs: list[RequestOutput] | None = None
        self.artifacts_path = artifacts_path

    def compose_dummy_prompts(self, prefix_max_len: int = 10_000, prompt_max_len: int = 20_000) -> None:
        prompts = list()
        targets = list()
        for datapoint in self.data_loader.data['train']:  # TODO: fix that somehow
            completion_lines = datapoint['completion_file'].split('\n')
            cumul_length = list()
            total_curr_length = 0
            for line in completion_lines:
                total_curr_length += len(line)
                cumul_length.append(total_curr_length)
            split_idx = len(completion_lines) // 2
            if cumul_length[split_idx] > prefix_max_len:
                # print(f'Changing split from line {split_idx} with {cumul_length[split_idx]} chars to ', end='\n\t---')
                length_condition = [cl <= prefix_max_len for cl in cumul_length]
                split_idx = max((i for i, x in enumerate(length_condition) if x), default=0)
                # print(f'line {split_idx} with {cumul_length[split_idx]} chars')
            file_level_prompt = '\n'.join(completion_lines[:split_idx])
            prompt = (datapoint['context'] + file_level_prompt)[-prompt_max_len:]
            prompts.append(prompt)
            targets.append('\n'.join(completion_lines[split_idx:]))
        self.prompts = prompts
        self.targets = targets

    def compose_random_prompts(self,
                               prefix_max_len: int = 10_000,
                               prompt_max_len: int = 40_000,
                               lines_per_datapoint: int = 10,
                               seed: int = 111
                               ) -> None:
        prompts = list()
        targets = list()
        rnd = random.Random(seed)
        for datapoint in self.data_loader.data['train']:  # TODO: fix that somehow
            added_lines_count = 0
            total_count = 0
            completion_lines = datapoint['completion_file'].split('\n')
            cumul_length = list()
            total_curr_length = 0
            for line in completion_lines:
                total_curr_length += len(line)
                cumul_length.append(total_curr_length)
            possible_split_idxs = [idx for idx, _ in enumerate(completion_lines) if cumul_length[idx] < prefix_max_len]
            possible_split_idxs = possible_split_idxs[10:]
            possible_split_idxs = [idx for idx in possible_split_idxs if 10 < len(completion_lines[idx].strip()) < 200]
            if len(possible_split_idxs) < lines_per_datapoint:
                sample_size = lines_per_datapoint
            else:
                sample_size = lines_per_datapoint
            split_idxs = rnd.sample(possible_split_idxs, sample_size)
            # print(split_idxs, [len(completion_lines[idx]) for idx in split_idxs])
            for split_idx in sorted(split_idxs):
                file_level_prompt = '\n'.join(completion_lines[:split_idx])
                prompt = (datapoint['context'] + file_level_prompt)[-prompt_max_len:]
                prompts.append(prompt)
                targets.append('\n'.join(completion_lines[split_idx:]))
        self.prompts = prompts
        self.targets = targets

    def get_outputs(self, **generation_params) -> None:
        self.outputs = self.inference_engine.generate(self.prompts,
                                                      sampling_params=self.sampling_params,
                                                      use_tqdm=True,
                                                      **generation_params
                                                      )
        # MEMORY CONSTRAINT WORKAROUND:
        # outputs = list()
        # from tqdm import tqdm
        # for prompt in tqdm(self.prompts):
        #     output = self.inference_engine.generate([prompt], sampling_params=self.sampling_params,)
        #     outputs.append(output[0])
        # self.outputs = outputs

    def save_results(self, filename: str | None = None) -> None:
        #
        # if self.prompts is None or self.outputs is None:
        #     raise ValueError('Prompts and Outputs must not be None')
        # if len(self.prompts) != len(self.outputs):
        #     raise ValueError('Prompts and Outputs must be the same length')
        # if self.targets is not None:
        #     if len(self.targets) != len(self.prompts) or len(self.targets) != len(self.outputs):
        #         raise ValueError('Targets must be the same length with Prompts and Outputs')
        #
        # if filename is None:
        #     filename = 'untitled_results.jsonl'
        # saving_path = os.path.join(self.artifacts_path, filename)
        # with jsonlines.open(saving_path, 'w') as writer:
        #     for idx in range(len(self.prompts)):
        #         result_dict = {
        #             'prompt': self.prompts[idx],
        #             'target': self.targets[idx] if self.targets[idx] else None,
        #             'output': self.outputs[idx].outputs[0].text,
        #             'cumulative_logprob': self.outputs[idx].outputs[0].cumulative_logprob,
        #         }
        #         writer.write(result_dict)

    def run(self):
        pass


if __name__ == '__main__':
    inference_engine = VllmEngine(hf_model_path='deepseek-ai/deepseek-coder-1.3b-base')
    sampling_params = {
        'temperature': 0.0,
        'min_tokens': 10,
    }
    data_loader = HFDataLoader(hf_dataset_path='ekaterina-blatova-jb/val_dataset_lora')
    evaluator = Evaluator(
        inference_engine=inference_engine,
        sampling_params=sampling_params,
        data_loader=data_loader
    )
    evaluator.compose_dummy_prompts(prefix_max_len=10_000, prompt_max_len=12_000)
    evaluator.get_outputs()
    evaluator.save_results(
        # filename='longer_results.jsonl'
    )
    # evaluator.compose_random_prompts(prefix_max_len=10_000, prompt_max_len=40_000)
    # print(len(evaluator.prompts))
