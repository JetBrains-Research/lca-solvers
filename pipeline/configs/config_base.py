from __future__ import annotations

import abc
from dataclasses import dataclass, asdict
from typing import TypeVar, Type

import yaml

T = TypeVar('T')


@dataclass
class ConfigBase(abc.ABC):
    dict = property(asdict)

    @classmethod
    @property
    @abc.abstractmethod
    def _default_path(cls) -> str:  # noqa: classmethod
        raise NotImplementedError

    @classmethod
    def from_yaml(cls: Type[T], path: str | None = None) -> T:
        if path is None:
            path = cls._default_path

        with open(path) as stream:
            return cls(**yaml.safe_load(stream))  # noqa: PyCharm bug?
