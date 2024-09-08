from pipeline.outputs.loggers.dummy_logger import DummyLogger
from pipeline.outputs.loggers.local_logger import LocalLogger
from pipeline.outputs.loggers.wandb_logger import WandbLogger

LOGGERS_REGISTRY = {
    'dummy_logger': DummyLogger,
    'local_logger': LocalLogger,
    'wandb_logger': WandbLogger,
}
