from __future__ import annotations
from dataclasses import dataclass, asdict

import yaml


@dataclass
class Config:
    dict = property(asdict)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as stream:
            return cls(**yaml.safe_load(stream))  # noqa: PyCharm bug?
