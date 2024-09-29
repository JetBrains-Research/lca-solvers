from pipeline.outputs.metrics.metric_base import MetricValue, OptimizationMode, MetricBase

from typing import Type

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


def top_k_accuracy_factory(k: int) -> Type[MetricBase]:
    class TopKAccuracy(MetricBase):
        mode = OptimizationMode.MAX

        def __init__(self) -> None:
            self.tp_plus_tn = 0
            self.num_tokens = 0

        @torch.inference_mode
        def micro_batch_update(self,
                               model_output: CausalLMOutputWithPast,
                               target_ids: torch.Tensor,
                               mask: torch.Tensor,
                               **_kwargs,
                               ) -> None:
            logits = model_output.logits[mask]
            target_ids = target_ids[mask]
            self.tp_plus_tn += (logits.topk(k, dim=-1).indices == target_ids.unsqueeze(-1)).any(-1).sum()
            self.num_tokens += mask.sum().item()

        def batch_commit(self, **_kwargs) -> MetricValue:
            batch_metric = float('nan') if not self.num_tokens else (self.tp_plus_tn / self.num_tokens)
            self.tp_plus_tn = 0
            self.num_tokens = 0
            return batch_metric

    return TopKAccuracy
