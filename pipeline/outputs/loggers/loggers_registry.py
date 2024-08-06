from pipeline.outputs.loggers.dummy_logger import DummyLogger
from pipeline.outputs.loggers.local_logger import LocalLogger
from pipeline.outputs.loggers.wandb_logger import WandbLogger

LOGGERS_REGISTRY = {
    'dummy': DummyLogger,
    'local': LocalLogger,
    'wandb': WandbLogger,
}
