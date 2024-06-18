from lca_eval_harness.data_classes.datapoint_base import DataPointBase
from lca_eval_harness.prompters.prompter_base import PrompterBase
from lca_eval_harness.prompters.prompter_data.prompter_output import PrompterOutput


class PrompterFileLevel(PrompterBase):
    def __init__(self,
                 **base_params,
                 # prefix_max_len: int = 10_000,
                 # prompt_max_len: int = 40_000,
                 ):
        base_params['is_cheating'] = False
        super().__init__(**base_params)
        self.identifier = 'file_level'

    def compose_prompt(self, datapoint: DataPointBase) -> list[PrompterOutput]:
        prompts = list()
        file_prefixes, targets = datapoint.split_completion_file()
        for file_prfx, raw_trgt in zip(file_prefixes, targets):
            trgt = raw_trgt if len(raw_trgt) > 0 else None
            prompts.append(PrompterOutput(prompt=file_prfx, target=trgt))
        return prompts


if __name__ == '__main__':
    print('base')
    prompter = PrompterBase()
    print('file-level')
    prompter_2 = PrompterFileLevel(is_cheating=False)
