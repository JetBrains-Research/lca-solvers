from pipeline.data.extension.line_classification.category_processors.category_processor_base import CategoryProcessorBase
from pipeline.data.extension.line_classification.category_processors.category_processor_dummy import CategoryProcessorDummy
from pipeline.data.extension.line_classification.category_processors.category_processor_lca_ce import CategoryProcessorCodeEngine

category_processors_registry = {
    'base': CategoryProcessorBase,
    'dummy': CategoryProcessorDummy,
    'code_engine': CategoryProcessorCodeEngine,
}