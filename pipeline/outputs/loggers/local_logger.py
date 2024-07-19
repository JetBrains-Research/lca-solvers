from pipeline.outputs.loggers.logger_base import Message, Log, LoggerBase
from pipeline.outputs.metrics.metric_base import MetricName, MetricValue

import csv
import json
import logging
import os
import sys
import traceback
import warnings
from types import TracebackType
from typing import NoReturn


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message_dict = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'content': record.msg,
        }

        indent = '    '
        json_string = json.dumps(message_dict, indent=4)
        json_string = indent.join(json_string.splitlines(keepends=True))
        json_string = indent + json_string

        return json_string


class JsonHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        if not os.path.exists(self.baseFilename) or os.stat(self.baseFilename).st_size == 0:
            self.stream.write('[\n')
        else:
            self.stream.seek(self.stream.tell() - 1)
            self.stream.truncate()
            self.stream.write(',\n')

        super().emit(record)

    def close(self) -> None:
        self.stream.seek(self.stream.tell() - 1)
        self.stream.write(']')

        super().close()


class LocalLogger(LoggerBase):
    def __init__(self,
                 train_csv: str,
                 valid_csv: str,
                 stdout_file: str,
                 stderr_file: str,
                 directory: str,
                 ) -> None:
        if train_csv == valid_csv:
            raise ValueError('The names of the train_csv and valid_csv files must be different.')

        train_csv, valid_csv, stdout_file, stderr_file = map(
            lambda x: os.path.join(directory, x),
            [train_csv, valid_csv, stdout_file, stderr_file],
        )

        self.train_csv = train_csv
        self.valid_csv = valid_csv

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = JsonFormatter()

        stdout_handler = JsonHandler(stdout_file)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)

        if stderr_file == stdout_file:
            stderr_handler = stdout_handler
        else:
            stderr_handler = JsonHandler(stderr_file)
            stderr_handler.setLevel(logging.WARNING)
            stderr_handler.setFormatter(formatter)
            stderr_handler.addFilter(lambda record: record.levelno >= logging.WARNING)

        self.logger.addHandler(stdout_handler)
        self.logger.addHandler(stderr_handler)

        warnings.showwarning = self.warning_handler
        sys.excepthook = self.exception_handler

    @staticmethod
    def write_metrics_to_csv(metrics: dict[MetricName, MetricValue], path: str) -> None:
        with open(path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)

    def log(self, metrics: Log) -> Log:
        iter_num = {'iter_num': metrics['iteration_number']}
        self.write_metrics_to_csv(iter_num | metrics['train_metrics'], self.train_csv)
        if 'valid_metrics' in metrics:
            self.write_metrics_to_csv(iter_num | metrics['valid_metrics'], self.valid_csv)
        return metrics

    def message(self, message: Message) -> Message:
        self.logger.info(message)
        return message

    def warning_handler(self, message: Warning, category: type, path: str, lineno: int, *_kwargs) -> None:
        self.logger.warning({
            'category': category.__name__,
            'location': f'{path}:{lineno}',
            'message': str(message),
        })

    def exception_handler(self, exc_type: type, exc_value: Exception, exc_traceback: TracebackType) -> NoReturn:
        if issubclass(exc_type, KeyboardInterrupt):
            self.message('Process was stopped due to a keyboard interrupt.')
        else:
            self.logger.error({
                'category': exc_type.__name__,
                'traceback': [{
                    'location': f'{filename}:{lineno} in {func_name}',
                    'line': line,
                } for filename, lineno, func_name, line in traceback.extract_tb(exc_traceback)],
                'message': str(exc_value),
            })
            self.message('Process finished with a non-zero exit code.')
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
