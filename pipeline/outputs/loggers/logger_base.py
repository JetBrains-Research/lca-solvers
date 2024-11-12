from pipeline.outputs.metrics.statistic_base import StatisticName, StatisticValue

from abc import ABC, abstractmethod
from typing import TypedDict, TypeVar, Type
from typing_extensions import NotRequired

T = TypeVar('T')
JsonAllowedTypes = dict | list | tuple | str | int | float | bool | None
Message = str | int | float | dict[str, JsonAllowedTypes]


class Log(TypedDict):  # TODO: replace with dataclass
    iteration_number: int
    train_metrics: NotRequired[dict[StatisticName, StatisticValue]]
    valid_metrics: NotRequired[dict[StatisticName, StatisticValue]]


class LoggerBase(ABC):
    _instance = None  # singleton pattern

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
