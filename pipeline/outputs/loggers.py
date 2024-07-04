import abc


class LoggerBase(abc.ABC):
    pass


class LocalLogger(LoggerBase):
    pass


class WandbLogger(LoggerBase):
    pass
