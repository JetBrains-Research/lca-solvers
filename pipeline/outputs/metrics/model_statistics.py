from pipeline.outputs.metrics.statistic_base import StatisticValue, StatisticBase

from typing import TypeVar

import torch

# avoiding cyclical imports
FullFineTuningTrainer = TypeVar('FullFineTuningTrainer')


class PastWeights(StatisticBase):
    def micro_batch_update(self, **_kwargs) -> None:
        pass

    @torch.inference_mode
    def batch_commit(self, trainer: FullFineTuningTrainer, **_kwargs) -> StatisticValue:
        past_weights_group = trainer.optimizer.param_groups[0]
        assert past_weights_group['name'] == 'past_weights'

        num_weights = 0
        weights_sum = 0
        for past_weight in past_weights_group['params']:
            num_weights += past_weight.numel()
            weights_sum += sum(past_weight.flatten().tolist())

        return weights_sum / num_weights
