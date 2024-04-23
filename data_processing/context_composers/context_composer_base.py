from data_processing.data_classes.datapoint_base import DatapointBase


class ContextComposerBase:
    def __init__(self):
        pass

    def compose_context(self, datapoint: DatapointBase, mode: str | None, max_length: int) -> str:
        raise NotImplementedError
