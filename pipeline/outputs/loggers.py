from abc import ABC


class LoggerBase(ABC):
    pass


class LocalLogger(LoggerBase):
    pass


class WandbLogger(LoggerBase):
    pass
