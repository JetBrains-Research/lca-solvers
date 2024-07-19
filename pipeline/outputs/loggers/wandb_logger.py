from pipeline.outputs.loggers.local_logger import LocalLogger
from pipeline.outputs.loggers.logger_base import Log

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
        wandb.init(*wandb_init_args, **wandb_init_kwargs)  # TODO: test resume

    def log(self, metrics: Log) -> Log:
        wandb_log = {'train': metrics['train_metrics']}
        if 'valid_metrics' in metrics:
            wandb_log['validation'] = metrics['valid_metrics']
        wandb.log(wandb_log)
        return super().log(metrics)
