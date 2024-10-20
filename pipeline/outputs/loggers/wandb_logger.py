from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.loggers.local_logger import LocalLogger
from pipeline.outputs.loggers.logger_base import Log

import wandb


class WandbLogger(LocalLogger):
    def __init__(self,
                 checkpointer: CheckpointManager,
                 train_csv: str,
                 valid_csv: str,
                 stdout_file: str,
                 stderr_file: str,
                 directory: str,
                 **wandb_init_kwargs,
                 ) -> None:
        super().__init__(train_csv, valid_csv, stdout_file, stderr_file, directory)
        wandb_init_kwargs['resume'] = wandb_init_kwargs.get('resume', checkpointer.get_wandb_resume_mode())
        wandb_init_kwargs['id'] = wandb_init_kwargs.get('id', wandb_init_kwargs['name'])
        wandb.init(**wandb_init_kwargs)

    def log(self, metrics: Log) -> Log:
        # TODO: nesting: don't forget the additional validation loop case

        if metrics['iteration_number'] <= self.last_logged_iter:
            return super().log(metrics)  # repeated iterations between checkpoints

        wandb_log = dict()
        if 'train_metrics' in metrics:
            wandb_log['train'] = metrics['train_metrics']
        if 'valid_metrics' in metrics:
            wandb_log['validation'] = metrics['valid_metrics']

        wandb.log(wandb_log, step=metrics['iteration_number'])
        return super().log(metrics)
