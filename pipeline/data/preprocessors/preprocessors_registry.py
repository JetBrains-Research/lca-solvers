from pipeline.data.preprocessors.completion_loss_preprocessor import CompletionLossPreprocessor
from pipeline.data.preprocessors.file_level_preprocessor import FileLevelPreprocessor
from pipeline.data.preprocessors.lm_preprocessor import LMPreprocessor
from pipeline.data.preprocessors.split_preprocessor import SplitPreprocessor


PREPROCESSORS_REGISTRY = {
    'completion_loss_preprocessor': CompletionLossPreprocessor,
    'file_level_preprocessor': FileLevelPreprocessor,
    'lm_preprocessor': LMPreprocessor,
    'split_preprocessor': SplitPreprocessor,
}
