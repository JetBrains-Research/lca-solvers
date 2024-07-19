from pipeline.outputs.metrics.metric_base import MetricName, MetricValue

from abc import ABC, abstractmethod
from typing import TypedDict, TypeVar, Type
from typing_extensions import NotRequired

T = TypeVar('T')
JsonAllowedTypes = dict | list | tuple | str | int | float | bool | None
Message = str | int | float | dict[str, JsonAllowedTypes]


class Log(TypedDict):
    iteration_number: int
    train_metrics: dict[MetricName, MetricValue]
    valid_metrics: NotRequired[dict[MetricName, MetricValue]]


class LoggerBase(ABC):
    _instance = None  # singleton pattern

    # TODO: test
    def __new__(cls: Type[T], *args, **kwargs) -> T:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @abstractmethod
    def log(self, metrics: Log) -> Log:
        raise NotImplementedError

    @abstractmethod
    def message(self, message: Message) -> Message:
        raise NotImplementedError
