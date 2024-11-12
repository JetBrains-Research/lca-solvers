from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.loggers.local_logger import LocalLogger
from pipeline.outputs.loggers.logger_base import Log

import time

import wandb


METRICS_NESTING = {
    'cross_entropy.decomposed': (
        'cross_entropy',
        'detached_cross_entropy',
        'completion_cross_entropy',
        'context_cross_entropy',
        'full_cross_entropy',
    ),
    'cross_entropy.categorized': (
        'InCommit_cross_entropy',
        'InFile_cross_entropy',
        'InProject_cross_entropy',
        'NonInformative_cross_entropy',
        'Other_cross_entropy',
        'OtherAPI_cross_entropy',
        'TODO_cross_entropy',
    ),
    'exact_match.decomposed': (
        'exact_match',
        'detached_exact_match',
        'completion_exact_match',
        'context_exact_match',
        'full_exact_match',
    ),
    'exact_match.categorized': (
        'InCommit_exact_match',
        'InFile_exact_match',
        'InProject_exact_match',
        'NonInformative_exact_match',
        'Other_exact_match',
        'OtherAPI_exact_match',
        'TODO_exact_match',
    ),
    'top_k_accuracy.categorized': (
        'InProject_top_1_accuracy',
        'InProject_top_3_accuracy',
        'InProject_top_5_accuracy',
        'InProject_top_10_accuracy',
    ),
    'statistics': (
        'epoch',
        'learning_rate',
    ),
}
METRICS_NESTING = {
    metric_name: group_name
    for group_name, group in METRICS_NESTING.items()
    for metric_name in group
}


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
        # wandb_init_kwargs['id'] = wandb_init_kwargs.get('id', wandb_init_kwargs['name'])
        wandb_init_kwargs['id'] = str(time.time())
        wandb.init(**wandb_init_kwargs)

    def log(self, metrics: Log) -> Log:
        if metrics['iteration_number'] <= self.last_logged_iter:
            return super().log(metrics)  # repeated iterations between checkpoints

        wandb_log = dict()

        if 'train_metrics' in metrics:
            for metric_name, metric_value in metrics['train_metrics'].items():
                if metric_name in METRICS_NESTING:
                    metric_name = f'train.{METRICS_NESTING[metric_name]}/{metric_name}'
                wandb_log[metric_name] = metric_value

        if 'valid_metrics' in metrics:
            for metric_name, metric_value in metrics['valid_metrics'].items():
                if metric_name in METRICS_NESTING:
                    metric_name = f'valid.{METRICS_NESTING[metric_name]}/{metric_name}'
                wandb_log[metric_name] = metric_value

        wandb.log(wandb_log, step=metrics['iteration_number'])
        return super().log(metrics)
