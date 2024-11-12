from pipeline.outputs.metrics.metric_base import OptimizationMode, MaskBasedMetric
from pipeline.outputs.metrics.statistic_base import StatisticName, StatisticValue

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


class TopKAccuracy(MaskBasedMetric):
    mode = OptimizationMode.MAX

    def __init__(self, k: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = k
        self.tp_plus_tn = 0
        self.num_tokens = 0

    @property
    def name(self) -> StatisticName:
        return super().name.replace('_k_', f'_{self.k}_')

    @torch.inference_mode
    def micro_batch_update(self,
                           model_output: CausalLMOutputWithPast,
                           target_ids: torch.Tensor,
                           **kwargs,
                           ) -> None:
        mask = self.get_mask(**kwargs)
        logits = model_output.logits[mask]
        target_ids = target_ids[mask].unsqueeze(-1)
        pred_ids = logits.topk(self.k, dim=-1).indices

        self.tp_plus_tn += (pred_ids == target_ids).any(-1).sum().item()
        self.num_tokens += mask.sum().item()

    def batch_commit(self, **_kwargs) -> StatisticValue:
        batch_metric = float('nan') if not self.num_tokens else (self.tp_plus_tn / self.num_tokens)
        self.tp_plus_tn = 0
        self.num_tokens = 0
        return batch_metric
