from cache_study.metric import Metrics
from cache_study.encoder import CacheEncoderMixin
from cache_study.generator import GeneratorMixin
from cache_study.preprocessor import PreprocessorMixin

from abc import ABC, abstractmethod
from typing import Any

import torch


class Pipeline(ABC):
    def eval(self, logits: list[torch.Tensor], target_ids: torch.Tensor) -> Metrics:
        pass  # TODO

    @abstractmethod
    def __call__(self, datapoint: dict[str, Any]) -> Metrics:
        raise NotImplementedError


class BaselinePipeline(Pipeline, PreprocessorMixin, GeneratorMixin):
    pass  # TODO


class TrialPipeline(Pipeline, PreprocessorMixin, CacheEncoderMixin, GeneratorMixin):
    pass  # TODO
