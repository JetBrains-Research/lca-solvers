from pipeline.outputs.metrics.metric_base import MetricValue, OptimizationMode, MetricBase

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


class CrossEntropy(MetricBase):
    mode = OptimizationMode.MIN

    def __init__(self) -> None:
        self.mean_loss = None
        self.num_tokens = None
        self.normalization = None
        self.reset()

    def reset(self) -> None:
        self.mean_loss = 0
        self.num_tokens = None
        self.normalization = 0

    @torch.inference_mode
    def micro_batch_update(self,
                           target_ids: torch.Tensor,
                           loss_mask: torch.Tensor,
                           model_output: CausalLMOutputWithPast,
                           loss: torch.Tensor | None = None,
                           **_kwargs) -> None:
        if loss is not None:
            # train
            loss_update = loss
            self.normalization += 1
        else:
            # validation
            loss_update = F.cross_entropy(model_output.logits[loss_mask], target_ids[loss_mask])
            self.normalization = 1

        loss_update = loss_update.item()
        num_tokens_update = loss_mask.count_nonzero().item()

        # loss correction w.r.t. number of masked tokens (for unbalanced batches)
        if self.num_tokens is None:
            self.mean_loss += loss_update
            self.num_tokens = 0
        else:
            tokens_ratio = num_tokens_update / self.num_tokens
            self.mean_loss += tokens_ratio * loss_update
            self.mean_loss /= tokens_ratio + 1

        self.num_tokens += num_tokens_update

    def batch_commit(self) -> MetricValue:
        batch_metric = self.mean_loss * self.normalization
        self.reset()
        return batch_metric


class DetachedCrossEntropy(CrossEntropy):
    def micro_batch_update(self, **kwargs) -> None:
        kwargs['loss_mask'] = ~kwargs['loss_mask']
        kwargs['loss'] = None
        return super().micro_batch_update(**kwargs)
