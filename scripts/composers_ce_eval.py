from pipeline.configs.checkpointer_config import CheckpointManagerConfig
from pipeline.configs.dataset_config import DatasetConfig
from pipeline.configs.model_config import ModelConfig
from pipeline.data.composers.blocks.file_filtering import *
from pipeline.data.composers.blocks.file_chunking import *
from pipeline.data.composers.blocks.chunk_ranking import *
from pipeline.data.composers.blocks.chunk_sorting import *
from pipeline.data.composers.blocks.chunk_harvesting import *
from pipeline.data.composers.chained_composer import ChainedComposer
from pipeline.data.preprocessors.completion_loss_preprocessor import CompletionLossPreprocessor
from pipeline.data.preprocessors.file_level_preprocessor import FileLevelPreprocessor
from pipeline.model.adapters.identity_adapter import IdentityAdapter
from pipeline.model.init import init_tokenizer, init_model
from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.checkpointers.data_structures import LoadingMode
from pipeline.trainers.validator import Validator

import jsonlines
from datasets import load_dataset

RANDOM_SEED = 1337
MODEL_NAME = 'deepseek-ai/deepseek-coder-1.3b-base'
RUNS = {
    'AbsoluteCalls_FullFT_DeepSeekCoder1p3Base_HP001': 'Absolute Calls with Path Distance sorting',
    'FileLevel_FullFT_DeepSeekCoder1p3Base_HP001': 'File-Level Completion',
    'FuncCallsWithStrip_FullFT_DeepSeekCoder1p3Base_HP001': 'Function Calls with Lines Strip',
    'LowTokensRatioFiltering_FullFT_DeepSeekCoder1p3Base_HP001': 'Low Token Ratio Filtering',
    'MediumLines_FullFT_DeepSeekCoder1p3Base_HP001': 'Low Token Ratio Filtering with Medium Line Length',
    'NearestDeclarations_FullFT_DeepSeekCoder1p3Base_HP001': 'Declarations Only sorted by Path Distance',
    'NoLongFiles_FullFT_DeepSeekCoder1p3Base_HP001': 'Long Files Filtering sorted by Function Calls',
    'NoLongFilesWithHalfMemory_FullFT_DeepSeekCoder1p3Base_HP001': 'Long Files Filtering with Half Memory',
    'PartialMemory_FullFT_DeepSeekCoder1p3Base_HP001': 'Path Distance Ordering with Half Memory',
    'PathDistance_FullFT_DeepSeekCoder1p3Base_HP002': 'Path Distance Ordering',
    'PathDistance_FullFT_DeepSeekCoder1p3Base_HP003': 'Path Distance without Completion Imports',
    'PureFuncCalls_FullFT_DeepSeekCoder1p3Base_HP001': 'Function Calls Ordering',
    'PythonFiles_FullFT_DeepSeekCoder1p3Base_HP001': 'Python Files Full Input Training',
    'RandomDeclarations_FullFT_DeepSeekCoder1p3Base_HP001': 'Declarations Only with Random Ordering',
    'RelativeCalls_FullFT_DeepSeekCoder1p3Base_HP001': 'Function Calls Ratio Ordering',
    'Strip_FullFT_DeepSeekCoder1p3Base_HP001': 'Stripped Filtered Lines sorted by Path Distance',
    'TextFiles_FullFT_DeepSeekCoder1p3Base_HP002': 'Text Files Groups',
    'No Fine-Tuning': 'No Fine-Tuning',
}
DATASET_CONFIGS = {
    'medium': 'configs/dataset/medium.yaml',
    'large': 'configs/dataset/large.yaml',
}
COMPOSER_KWARGS = dict(
    pre_context_prompt='# {}\n',
    post_context_prompt='\n\n',
    path_comment_template='# {filename}\n{content}',
    recalculate_random_category=True,
)
COMPOSERS = {
    'File level': ChainedComposer([
        FileGrainedChunker(),
        PathCommentHarvester(chunks_sep='\n\n', path_comment_template='# {filename}\n{content}'),
    ], **COMPOSER_KWARGS),
    'Path distance': ChainedComposer([
        FileLengthFilter(min_len=10, max_len=10_000),
        FileGrainedChunker(),
        NegativePathDistanceRanker(),
        LexicographicSorter(),
        PathCommentHarvester(chunks_sep='\n\n', path_comment_template='# {filename}\n{content}'),
    ], **COMPOSER_KWARGS),
    'Python files with mixed ordering': ChainedComposer([
        InclusiveFileExtensionFilter(whitelist=['.py']),
        FileLengthFilter(min_len=10, max_len=10_000),
        FileGrainedChunker(),
        NegativePathDistanceRanker(),
        RandomRanker(random_seed=RANDOM_SEED),
        MixedSorter(),
        PathCommentHarvester(chunks_sep='\n\n', path_comment_template='# {filename}\n{content}'),
    ], **COMPOSER_KWARGS),
    'Python and text files with mixed ordering': ChainedComposer([
        InclusiveFileExtensionFilter(whitelist=['.py', '.md', '.txt', '.rst']),
        FileLengthFilter(min_len=10, max_len=10_000),
        FileGrainedChunker(),
        NegativePathDistanceRanker(),
        RandomRanker(random_seed=RANDOM_SEED),
        MixedSorter(),
        PathCommentHarvester(chunks_sep='\n\n', path_comment_template='# {filename}\n{content}'),
    ], **COMPOSER_KWARGS),
    'Python code chunks with mixed ordering': ChainedComposer([
        InclusiveFileExtensionFilter(whitelist=['.py']),
        FileLengthFilter(min_len=10, max_len=10_000),
        CodeOnlyChunker(),
        NegativePathDistanceRanker(),
        RandomRanker(random_seed=RANDOM_SEED),
        MixedSorter(),
        PathCommentHarvester(chunks_sep='\n\n', path_comment_template='# {filename}\n{content}'),
    ], **COMPOSER_KWARGS),
}
OUTPUT_FILE = 'extra/outputs/composers_ce_eval/checkpoint_comparison.jsonl'


def main() -> None:
    checkpointer_config = CheckpointManagerConfig(
        init_from=LoadingMode.SCRATCH,
        main_metric='cross_entropy',
        directory=...)
    checkpointer = CheckpointManager(**checkpointer_config.dict)

    valid_metrics = checkpointer.init_metrics('valid_metrics', [
        'cross_entropy',
        'detached_cross_entropy',
        'completion_cross_entropy',
        'context_cross_entropy',
        'full_cross_entropy',
        'commited_cross_entropy',
        'common_cross_entropy',
        'infile_cross_entropy',
        'inproject_cross_entropy',
        'non_informative_cross_entropy',
        'random_cross_entropy'])
    checkpointer.init_from = LoadingMode.BEST

    model_config = ModelConfig(
        tokenizer_name=MODEL_NAME,
        model_name=MODEL_NAME,
        trust_remote_code=True,
        load_from=...,
        compile=False)
    tokenizer = init_tokenizer(model_config)
    adapter = IdentityAdapter(MODEL_NAME, params_pattern=None)

    preprocessor_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=16384,
        context_tokens=12288,
        loss_ratio=1,
        num_chars_per_token=6,
        use_sep_token=False,
        padding=True,
        verbose=False)
    completion_loss_preprocessor = CompletionLossPreprocessor(**preprocessor_kwargs)
    file_level_preprocessor = FileLevelPreprocessor(**preprocessor_kwargs)

    for run_name in RUNS:
        if run_name == 'No Fine-Tuning':
            checkpointer.init_from = LoadingMode.SCRATCH
        else:
            checkpointer.init_from = LoadingMode.BEST

        checkpointer.directory = os.path.join('runs', run_name, 'checkpoints')
        model_config.load_from = checkpointer.get_model_subdirectory()

        model = init_model(model_config).eval()
        model.requires_grad_(False)

        for ds_name, ds_config in DATASET_CONFIGS.items():
            ds_config = DatasetConfig.from_yaml(ds_config)
            dataset = load_dataset(**ds_config.dict)

            for composer_name, composer in COMPOSERS.items():
                if composer_name == 'File level':
                    preprocessor = file_level_preprocessor
                else:
                    preprocessor = completion_loss_preprocessor

                transform = lambda x: preprocessor(composer.compose_batch(x))
                dataset.set_transform(transform)

                validator = Validator(
                    model=model,
                    adapter=adapter,
                    valid_metrics=valid_metrics,
                    valid_ds=dataset,
                    batch_size=1,
                    num_workers=8,
                    prefetch_factor=8)
                result = {
                    'run_name': run_name,
                    'composer': composer_name,
                    'dataset': ds_name,
                } | validator.validate()

                with jsonlines.open(OUTPUT_FILE, mode='a') as writer:
                    writer.write(result)


if __name__ == '__main__':
    main()
