from dataclasses import dataclass


@dataclass
class ModelCheckpointConfig:
    freq: int
    directory: str
    init_from: str | None = None


@dataclass
class TopKModelCheckpointConfig:
    pass
