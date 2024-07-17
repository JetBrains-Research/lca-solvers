from pipeline.outputs.loggers.logger_base import Message, Log, LoggerBase


class DummyLogger(LoggerBase):
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def log(self, metrics: Log) -> Log:
        return metrics

    def message(self, message: Message) -> Message:
        return message
