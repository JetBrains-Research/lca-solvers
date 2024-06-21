import warnings

from lca_eval_harness.data_classes.datapoint_base import DataPointBase
from lca_eval_harness.prompters.prompter_data.prompter_output import PrompterOutput


class PrompterBase:
    def __init__(self,
                 # prefix_max_len: int = 10_000,
                 prompt_max_len: int = 40_000,
                 is_cheating: bool = True):
        self.identifier = 'base'
        # self.prefix_max_len = prefix_max_len
        self.prompt_max_len = prompt_max_len
        self.is_cheating = is_cheating
        self._cheating_warning()

    def compose_prompt(self, datapoint: DataPointBase) -> list[PrompterOutput]:
        raise NotImplementedError

    def _cheating_warning(self):
        if self.is_cheating:
            warnings.warn("Your prompter is cheating. Change to something else if it is not intentional")
