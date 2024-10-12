from pipeline.data.categories import CATEGORY2ID
from pipeline.outputs.metrics.statistic_base import StatisticName, StatisticBase

from abc import ABC, abstractmethod
from enum import Enum

import torch


class OptimizationMode(str, Enum):
    MIN = 'minimization'
    MAX = 'maximization'


class MetricBase(StatisticBase, ABC):
    @property
    @abstractmethod
    def mode(self) -> OptimizationMode:
        raise NotImplementedError


class MaskType(str, Enum):
    ATTACHED = ''
    DETACHED = 'detached'
    COMPLETION = 'completion'
    CONTEXT = 'context'
    FULL = 'full'
    COMMITED = 'commited'
    COMMON = 'common'
    INFILE = 'infile'
    INPROJECT = 'inproject'
    NON_INFORMATIVE = 'non_informative'
    RANDOM = 'random'
    OTHER = 'other'


class MaskBasedMetric(MetricBase, ABC):
    def __init__(self, mask_type: MaskType) -> None:
        self.mask_type = mask_type

    @property
    def name(self) -> StatisticName:
        prefix = '' if self.mask_type == MaskType.ATTACHED else f'{self.mask_type}_'
        return prefix + super().name

    def get_mask(self, **kwargs) -> torch.Tensor:
        match self.mask_type:
            case MaskType.ATTACHED:
                return kwargs['loss_mask']
            case MaskType.DETACHED:
                return ~kwargs['loss_mask'] & kwargs['target_attn_mask']
            case MaskType.COMPLETION:
                return kwargs['completion_mask']
            case MaskType.CONTEXT:
                return ~kwargs['completion_mask'] & kwargs['target_attn_mask']
            case MaskType.FULL:
                return kwargs['target_attn_mask']
            case _:  # categorized
                return kwargs['category_ids'] == CATEGORY2ID[self.mask_type]
