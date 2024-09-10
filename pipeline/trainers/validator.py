from pipeline.model.adapters.adapter_base import AdapterBase
from pipeline.outputs.metrics.metric_base import MetricName, MetricValue, MetricBase
from pipeline.trainers.trainer_base import TrainerBase

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel


class Validator(TrainerBase):
    def __init__(self,
                 model: PreTrainedModel,
                 adapter: AdapterBase,
                 valid_metrics: dict[MetricName, MetricBase],
                 valid_ds: Dataset,
                 batch_size: int,
                 num_workers: int,
                 prefetch_factor: int,
                 ) -> None:
        self.model = model
        self.adapter = adapter
        self.valid_metrics = valid_metrics
        self.valid_dl = DataLoader(
            dataset=valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(model.device.type == 'cuda'),
            drop_last=False,
            prefetch_factor=prefetch_factor,
            pin_memory_device=str(model.device),
        )

    @torch.inference_mode
    def validate(self, verbose: bool = True) -> dict[MetricName, MetricValue]:
        training = self.model.training
        self.model.eval()

        valid_iter = tqdm(
            iterable=self.valid_dl,
            desc='Validation steps',
            leave=None,
            disable=not verbose,
        )

        for batch in valid_iter:
            inputs = (
                input_ids, target_ids,
                loss_mask, completion_mask, category_ids,
                input_attn_mask, target_attn_mask,
            ) = tuple(t.to(self.model.device) for t in batch.values())
            args, kwargs = self.adapter.get_args_kwargs(*inputs)

            model_output = self.model(*args, **kwargs)
            loss_per_token = F.cross_entropy(
                input=model_output.logits.flatten(0, 1),
                target=target_ids.flatten(0, 1),
                reduction='none',
            ).view_as(target_ids)

            locals_copy = locals().copy()
            locals_copy['trainer'] = locals_copy.pop('self')
            [metric.micro_batch_update(**locals_copy) for metric in self.valid_metrics.values()]
            del locals_copy

        valid_log = {name: metric.batch_commit() for name, metric in self.valid_metrics.items()}

        self.model.train(training)
        return valid_log

    def train(self, *_args, **_kwargs) -> None:
        raise NotImplementedError
