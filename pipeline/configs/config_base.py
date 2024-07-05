from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Self

import abc

import yaml


@dataclass
class ConfigBase(abc.ABC):
    dict = property(asdict)

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        with open(path) as stream:
            return cls(**yaml.safe_load(stream))  # noqa: PyCharm bug?
