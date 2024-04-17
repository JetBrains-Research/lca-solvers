from data_classes.datapoint_base import DatapointBase


class ContextComposerBase:
    def __init__(self):
        pass

    def compose_context(self, datapoint: DatapointBase) -> str:
        raise NotImplementedErro