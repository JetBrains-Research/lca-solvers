from cache_study.encoder import CacheEncoderMixin
from cache_study.generator import GeneratorMixin
from cache_study.preprocessor import BOSUsage, PreprocessorMixin
from composers.chain.chain import UnsafeComposerChain

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


@dataclass
class Metrics:
    cross_entropy: list[float] = field(default_factory=list)
    exact_match: list[float] = field(default_factory=list)


class Pipeline(ABC):
    @torch.inference_mode
    def eval(self, logits: torch.Tensor, target_ids: torch.Tensor) -> tuple[float, float]:
        loss = F.cross_entropy(logits, target_ids).item()
        return loss, -1  # TODO: add Exact Match

    @abstractmethod
    def __call__(self, datapoint: dict[str, Any]) -> Metrics:
        raise NotImplementedError


class BaselinePipeline(Pipeline, PreprocessorMixin, GeneratorMixin):
    def __init__(self,
                 composer: UnsafeComposerChain,
                 tokenizer: PreTrainedTokenizerBase,
                 bos_usage: BOSUsage,
                 full_model: nn.Module,
                 ) -> None:
        if bos_usage == BOSUsage.TAIL_SEAM:
            raise ValueError('Tail seaming is only possible for TrialPipeline.')

        PreprocessorMixin.__init__(self, composer, tokenizer, bos_usage)
        GeneratorMixin.__init__(self, full_model, hs_input=False)

    def __call__(self, datapoint: dict[str, Any]) -> Metrics:
        head_blocks, tail_blocks, target_ids = self.preprocess(datapoint, self.model.device)
        metrics = Metrics()

        for logits in self.produce_logits(head_blocks, tail_blocks, len(target_ids)):
            loss, em = self.eval(logits, target_ids)
            metrics.cross_entropy.append(loss)
            metrics.exact_match.append(em)

        return metrics


class TrialPipeline(Pipeline, PreprocessorMixin, CacheEncoderMixin, GeneratorMixin):
    pass  # TODO
