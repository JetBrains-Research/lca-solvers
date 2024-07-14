from pipeline.outputs.metrics.metric_base import MetricName, MetricValue

from abc import ABC, abstractmethod
from typing import TypeVar, Type

T = TypeVar('T')
JsonAllowedTypes = dict | list | tuple | str | int | float | bool | None


class LoggerBase(ABC):
    _instance = None  # singleton pattern

    # TODO: test
    def __new__(cls: Type[T], *args, **kwargs) -> T:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @abstractmethod
    def train_log(self, metrics: dict[MetricName, MetricValue]) -> dict[MetricName, MetricValue]:
        raise NotImplementedError

    @abstractmethod
    def valid_log(self, metrics: dict[MetricName, MetricValue]) -> dict[MetricName, MetricValue]:
        raise NotImplementedError

    @abstractmethod
    def message(self, message: str | dict[str, JsonAllowedTypes]) -> str | dict[str, JsonAllowedTypes]:
        raise NotImplementedError
