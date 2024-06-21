from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    @property
    @abstractmethod
    def data(self):
        pass
