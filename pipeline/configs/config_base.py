from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import TypeVar

import abc

import yaml

T = TypeVar('T')


@dataclass
class ConfigBase(abc.ABC):
    dict = property(asdict)

    @classmethod
    def from_yaml(cls: T, path: str) -> T:
        with open(path) as stream:
            return cls(**yaml.safe_load(stream))  # noqa: PyCharm bug?
