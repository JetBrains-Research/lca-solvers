from pipeline.data.composed_datapoint import BatchComposedDatapoint

import abc

from transformers import BatchEncoding


class PreprocessorBase(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch: BatchComposedDatapoint) -> BatchEncoding:
        raise NotImplementedError
