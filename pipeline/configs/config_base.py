from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import TypeVar, Type

import yaml

T = TypeVar('T')


@dataclass
class ConfigBase(ABC):
    dict = property(asdict)

    @property
    @abstractmethod
    def _default_path(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_yaml(cls: Type[T], path: str | None = None) -> T:
        if path is None:
            path = cls._default_path

        with open(path) as stream:
            return cls(**yaml.safe_load(stream))  # noqa: PyCharm bug?
