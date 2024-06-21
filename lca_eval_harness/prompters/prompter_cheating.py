from lca_eval_harness.data_classes.datapoint_base import DataPointBase
from lca_eval_harness.prompters.prompter_base import PrompterBase
from lca_eval_harness.prompters.prompter_data.prompter_output import PrompterOutput


class PrompterCheating(PrompterBase):
    def __init__(self,
                 **base_params,
                 ):
        base_params['is_cheating'] = True
        super().__init__(**base_params)
        self.identifier = 'cheating'

    def compose_prompt(self, datapoint: DataPointBase) -> list[PrompterOutput]:
        prompts = list()
        file_prefixes, targets = datapoint.split_completion_file()
        for file_prefix, raw_target in zip(file_prefixes, targets):
            target = raw_target if len(raw_target) > 0 else None
            prompt = file_prefix[-300:] + raw_target[:300]
            prompt = prompt[self.prompt_max_len:]
            prompts.append(PrompterOutput(prompt=prompt, target=target))
        return prompts
