from pipeline.data.composed_datapoint import BatchedComposedDatapoint

import abc

from transformers import BatchEncoding


class PreprocessorBase(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch: BatchedComposedDatapoint) -> BatchEncoding:
        raise NotImplementedError
