from pipeline.outputs.metrics.metric_base import OptimizationMode, MaskBasedMetric
from pipeline.outputs.metrics.statistic_base import StatisticValue

import torch
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast


class ExactMatch(MaskBasedMetric):
    mode = OptimizationMode.MAX
    requires_tokenizer = True

    def __init__(self, tokenizer: PreTrainedTokenizerBase, min_tokens: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if min_tokens < 1:
            raise ValueError('min_tokens must be a natural number.')

        self.vocab_size = len(tokenizer)
        self.list_newline_ids = [
            token_id for token_id in range(self.vocab_size)
            if '\n' in tokenizer.decode(token_id)]
        self.set_newline_ids = set(self.list_newline_ids)

        self.min_tokens = min_tokens
        self.num_matches = 0
        self.num_lines = 0

    @torch.inference_mode
    def micro_batch_update(self,
                           model_output: CausalLMOutputWithPast,
                           target_ids: torch.Tensor,
                           **kwargs,
                           ) -> None:
        mask = self.get_mask(**kwargs)
        logits = model_output.logits[mask]
        logits[:, self.vocab_size:] = -torch.inf
        target_ids = target_ids[mask].tolist()

        gt_line_ids = [[]]

        for token_id in target_ids:
            gt_line_ids[-1].append(token_id)
            if token_id in self.set_newline_ids:
                gt_line_ids.append([])

        if not gt_line_ids[-1]:
            gt_line_ids.pop(-1)

        start_idx = 0
        for line in gt_line_ids:
            line_length = len(line)
            line_logits = logits[start_idx:(start_idx + line_length)]
            pred_line = line_logits.argmax(-1).tolist()

            short_line = (line_length < self.min_tokens)
            interruption = bool(self.set_newline_ids & set(pred_line[:(self.min_tokens - 1)]))

            if not short_line and interruption:  # give the model a second chance
                line_logits[:(self.min_tokens - 1), self.list_newline_ids] = -torch.inf
                pred_line = line_logits.argmax(-1).tolist()

            start_idx += line_length

            self.num_matches += (pred_line == line)
        self.num_lines += len(gt_line_ids)

    def batch_commit(self, **_kwargs) -> StatisticValue:
        batch_metric = float('nan') if not self.num_lines else (self.num_matches / self.num_lines)
        self.num_matches = 0
        self.num_lines = 0
        return batch_metric
