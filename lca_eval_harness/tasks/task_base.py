import dataclasses
import os
import warnings
from typing import Any

import jsonlines

from lca_eval_harness.dataset_loaders.base_data_loader import BaseDataLoader
from lca_eval_harness.model_inference.base_engine import BaseEngine, BaseEngineOutput
from lca_eval_harness.prompters.prompter_base import PrompterBase


@dataclasses.dataclass
class TaskConfig:
    generation_engine: BaseEngine
    prompter: PrompterBase
    data_loader: BaseDataLoader
    artifacts_dir_path: str
    sampling_params: dict | None = None
    results_filename: str = 'untitled_results.jsonl'


class TaskBase:
    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config
        self.engine = task_config.generation_engine
        self.prompter = task_config.prompter
        self.prompter_identifier = task_config.prompter.identifier
        self.data_loader = task_config.data_loader
        self.sampling_params = self._get_sampling_params(task_config.sampling_params)
        self.prompts: list[str] | None = None
        self.targets: list[str] | None = None
        self.outputs: list[BaseEngineOutput] | None = None
        self.artifacts_dir_path = task_config.artifacts_dir_path
        self.results_filename = task_config.results_filename

    def compose_prompts(self) -> None:
        prompts = list()
        targets = list()
        for datapoint in self.data_loader.data:
            prompt_outputs = self.prompter.compose_prompt(datapoint)
            for prompt_output in prompt_outputs:
                prompts.append(prompt_output.prompt)
                targets.append(prompt_output.target)
        self.prompts = prompts
        self.targets = targets

    def _get_sampling_params(self, sampling_params: dict | None) -> Any:
        if sampling_params is None:
            return None
        return self.engine.get_sampling_params(**sampling_params)

    def get_outputs(self, **generation_params) -> None:
        if self.sampling_params is not None:
            if 'sampling_params' in generation_params:
                warnings.warn(
                    'Sampling Params provided both in argument and in task_config, '
                    'will override with task_config sampling'
                )
            generation_params['sampling_params'] = self.sampling_params

        self.outputs = self.engine.generate(self.prompts, **generation_params)

    def save_results(self, filename: str | None = None) -> None:
        if self.prompts is None or self.outputs is None:
            raise ValueError('Prompts and Outputs must not be None')
        if len(self.prompts) != len(self.outputs):
            raise ValueError('Prompts and Outputs must be the same length')
        if self.targets is not None:
            if len(self.targets) != len(self.prompts) or len(self.targets) != len(self.outputs):
                raise ValueError('Targets must be the same length with Prompts and Outputs')

        self._resolve_directories(filename)
        if filename is None:
            filename = self.results_filename
        saving_path = os.path.join(self.artifacts_dir_path, filename)
        with jsonlines.open(saving_path, 'w') as writer:
            for idx in range(len(self.prompts)):
                result_dict = {
                    'prompt': self.prompts[idx],
                    'target': self.targets[idx] if self.targets[idx] else None,
                    'output': self.outputs[idx].get_out_text(),
                    'prompter': self.prompter_identifier,
                    **self.outputs[idx].get_out_extras()
                }
                writer.write(result_dict)

    def _resolve_directories(self, filename: str | None = None) -> None:
        if filename is None:
            filename = self.results_filename
        full_path = os.path.join(self.artifacts_dir_path, filename)
        dir_path = os.path.dirname(full_path)
        os.makedirs(dir_path, exist_ok=True)

    def run(self, save_results: bool = True, **generation_params) -> None:
        self.compose_prompts()
        self.get_outputs(**generation_params)
        if save_results:
            self.save_results()


if __name__ == '__main__':
    from lca_eval_harness.tasks.dev_task_config import dev_task_config
    task = TaskBase(dev_task_config)
    task.run()
