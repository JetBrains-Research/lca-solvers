from pipeline.outputs.loggers.local_logger import LocalLogger
from pipeline.outputs.metrics.metric_base import MetricName, MetricValue

import wandb


class WandbLogger(LocalLogger):
    def __init__(self,
                 train_csv: str,
                 valid_csv: str,
                 stdout_file: str,
                 stderr_file: str,
                 directory: str,
                 *wandb_init_args,
                 **wandb_init_kwargs,
                 ) -> None:
        super().__init__(train_csv, valid_csv, stdout_file, stderr_file, directory)
        wandb.init(*wandb_init_args, **wandb_init_kwargs)

    def train_log(self, metrics: dict[MetricName, MetricValue]) -> dict[MetricName, MetricValue]:
        super().train_log(metrics)
        wandb.log({'train': metrics})
        return metrics

    def valid_log(self, metrics: dict[MetricName, MetricValue]) -> dict[MetricName, MetricValue]:
        super().train_log(metrics)
        wandb.log({'validation': metrics}, commit=False)
        return metrics
