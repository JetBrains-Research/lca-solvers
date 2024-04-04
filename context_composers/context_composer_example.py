import random

from context_composers.context_composer_base import ContextComposerBase
from data_classes.datapoint_base import DatapointBase


class ContextComposerExample(ContextComposerBase):
    def __init__(self):
        super().__init__()
        pass

    def compose_context(self, datapoint: DatapointBase, mode: str | None = None, max_length: int = -1) -> str:
        relevant_context = datapoint.get_relevant_context()
        non_relevant_context = datapoint.get_non_relevant_context()
        filename_0 = random.choice(list(non_relevant_context))
        content_0 = non_relevant_context[filename_0]
        filename_1 = random.choice(list(relevant_context))
        content_1 = relevant_context[filename_1]
        compl_filename = datapoint.get_completion_filenames()[0]
        context = f'# {filename_0}\n\n{content_0}\n\n# {filename_1}\n\n{content_1}\n\n# {compl_filename}'
        if max_length > 0:
            context = context[-max_length:]

        return context

