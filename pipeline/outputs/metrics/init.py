from pipeline.outputs.metrics.statistic_base import StatisticName, StatisticBase
from pipeline.outputs.metrics.metrics_registry import METRICS_REGISTRY

import os
from typing import Iterable

import yaml
from transformers import PreTrainedTokenizerBase


def init_metrics(loaded_config: Iterable[StatisticName],
                 configs_dir: str,
                 tokenizer: PreTrainedTokenizerBase,
                 ) -> list[StatisticBase]:
    # TODO: additional validation loop is not considered
    metrics = list()

    for path in loaded_config:
        full_path = os.path.join(configs_dir, 'metrics/metrics', path)
        metric_name = os.path.basename(os.path.dirname(path))

        with open(full_path) as stream:
            metric_config = yaml.safe_load(stream)

        if metric_config is None:
            metric_config = dict()

        metric_cls = METRICS_REGISTRY[metric_name]
        if metric_cls.requires_tokenizer:
            metric_config['tokenizer'] = tokenizer

        metric = metric_cls(**metric_config)
        metrics.append(metric)

    return metrics
