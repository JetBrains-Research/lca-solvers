from dataclasses import asdict, dataclass, fields
from pprint import pformat
from typing import Any, TypeVar, Type

import yaml

T = TypeVar('T')


@dataclass
class ConfigBase:
    def __str__(self) -> str:
        return pformat(self)

    @classmethod
    def from_dict(cls: Type[T], dictionary: dict[str, Any]) -> T:
        config_fields = set(field.name for field in fields(cls))
        kwargs = {key: value for key, value in dictionary.items() if key in config_fields}
        return cls(**kwargs)  # noqa: PyCharm bug?

    @classmethod
    def from_yaml(cls: Type[T], path: str | None = None) -> T:
        if path is None:
            path = cls._default_path

        with open(path) as stream:
            return cls(**yaml.safe_load(stream))  # noqa: PyCharm bug?

    dict = property(vars)
