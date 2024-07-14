from pipeline.outputs.loggers.logger_base import JsonAllowedTypes, LoggerBase
from pipeline.outputs.metrics.metric_base import MetricName, MetricValue


class DummyLogger(LoggerBase):
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def train_log(self, metrics: dict[MetricName, MetricValue]) -> dict[MetricName, MetricValue]:
        return metrics

    def valid_log(self, metrics: dict[MetricName, MetricValue]) -> dict[MetricName, MetricValue]:
        return metrics

    def message(self, message: str | dict[str, JsonAllowedTypes]) -> str | dict[str, JsonAllowedTypes]:
        return message
