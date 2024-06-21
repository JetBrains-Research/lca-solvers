from lca_eval_harness.data_classes.datapoint_base import DataPointBase
from lca_eval_harness.prompters.prompter_base import PrompterBase
from lca_eval_harness.prompters.prompter_data.prompter_output import PrompterOutput


class PrompterLongerLines(PrompterBase):
    def __init__(self,
                 **base_params,
                 # prefix_max_len: int = 10_000,
                 # prompt_max_len: int = 40_000,
                 ):
        base_params['is_cheating'] = False
        super().__init__(**base_params)
        self.identifier = 'longer_lines'

    def compose_prompt(self, datapoint: DataPointBase, **filter_args) -> list[PrompterOutput]:
        prompts = list()
        file_prefixes, targets = datapoint.split_completion_file()
        for file_prefix, raw_target in zip(file_prefixes, targets):
            target = raw_target if len(raw_target) > 0 else None
            context = datapoint.context
            context = self._filter_short_lines(context, **filter_args)
            prompt = context + file_prefix
            prompt = prompt[-self.prompt_max_len:]
            prompts.append(PrompterOutput(prompt=prompt, target=target))
        return prompts

    @staticmethod
    def _filter_short_lines(code: str, min_line_len: int = 10) -> str:
        lines = code.split('\n')
        filtered_lines = [line for line in lines if len(line) >= min_line_len]
        filtered_code = '\n'.join(filtered_lines)
        return filtered_code
