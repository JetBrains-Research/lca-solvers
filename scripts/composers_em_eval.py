# kotlineval==0.1.6
from kotlineval.data.plcc.data_loader import get_dataloader
from kotlineval.eval.plcc.evaluator import Evaluator
from kotlineval.eval.vllm_engine import VllmEngine

from pipeline.configs.checkpointer_config import CheckpointManagerConfig
from pipeline.data.composers.init import init_composer
from pipeline.data.composers.chained_composer import ChainedComposer
from pipeline.data.composers.blocks.file_filtering import InclusiveFileExtensionFilter, EmptyFileFilter
from pipeline.data.composers.blocks.file_chunking import FileGrainedChunker
from pipeline.data.composers.blocks.chunk_ranking import NegativePathDistanceRanker
from pipeline.data.composers.blocks.chunk_sorting import LexicographicSorter
from pipeline.data.composers.blocks.chunk_harvesting import PathCommentHarvester
from pipeline.environment.hardware import get_free_device
from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.checkpointers.data_structures import LoadingMode

import os

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer


MODEL_NAME = 'deepseek-ai/deepseek-coder-1.3b-base'
RUNS = [
    'AbsoluteCalls_FullFT_DeepSeekCoder1p3Base_HP001',
    'FileLevel_FullFT_DeepSeekCoder1p3Base_HP001',
    'FuncCallsWithStrip_FullFT_DeepSeekCoder1p3Base_HP001',
    'LowTokensRatioFiltering_FullFT_DeepSeekCoder1p3Base_HP001',
    'MediumLines_FullFT_DeepSeekCoder1p3Base_HP001',
    'NearestDeclarations_FullFT_DeepSeekCoder1p3Base_HP001',
    'NoLongFiles_FullFT_DeepSeekCoder1p3Base_HP001',
    'NoLongFilesWithHalfMemory_FullFT_DeepSeekCoder1p3Base_HP001',
    'PartialMemory_FullFT_DeepSeekCoder1p3Base_HP001',
    'PathDistance_FullFT_DeepSeekCoder1p3Base_HP002',
    'PathDistance_FullFT_DeepSeekCoder1p3Base_HP003',
    'PureFuncCalls_FullFT_DeepSeekCoder1p3Base_HP001',
    'PythonFiles_FullFT_DeepSeekCoder1p3Base_HP001',
    'RandomDeclarations_FullFT_DeepSeekCoder1p3Base_HP001',
    'RelativeCalls_FullFT_DeepSeekCoder1p3Base_HP001',
    'Strip_FullFT_DeepSeekCoder1p3Base_HP001',
    'TextFiles_FullFT_DeepSeekCoder1p3Base_HP002',
]
RUNS_DIR = 'runs'
CONFIGS_DIR = 'configs'


class EmbeddedComposer(ChainedComposer):
    def context_and_completion_composer(self, datapoint: dict, line_index: int) -> dict[str, str]:
        for category, line_ids in datapoint['completion_lines'].items():
            if line_index in line_ids:
                line_category = category
                line_idx_position = line_ids.index(line_index)
                break
        else:
            raise ValueError('Invalid line index.')

        datapoint_copy = datapoint.copy()
        datapoint_copy.pop('scope')
        composed_datapoint = self.compose(datapoint_copy)

        line_index = composed_datapoint['completion_lines'][line_category][line_idx_position]
        completion_content = composed_datapoint['composed_completion'].split('\n')
        completion_filename = datapoint['completion_file']['filename']
        project_context = composed_datapoint['composed_context']

        gt = completion_content[line_index]
        prefix = '\n'.join(completion_content[:line_index])
        postfix = '\n'.join(completion_content[line_index + 1:])
        full_context = (project_context + prefix).strip() + '\n'

        return {
            'gt': gt,
            'prefix': prefix,
            'postfix': postfix,
            'file_context': prefix,
            'filename': completion_filename,
            'full_context': full_context,
        }


def run_eval_plcc(config: DictConfig, composer: EmbeddedComposer) -> None:
    config.model.model_name, true_model = MODEL_NAME, config.model.model_name
    dataloader = get_dataloader(config, composer)
    config.model.model_name = true_model
    generation_engine = VllmEngine(
        config.model.model_name,
        context_size=config.eval.context_size,
        vllm_args=dict(
            tokenizer=MODEL_NAME,
        ),
        generation_args=dict(config.vllm.generation_args),
    )
    evaluator = Evaluator(
        engine=generation_engine,
        result_folder=config.output.result_folder,
        result_filename=config.output.results_filename,
    )
    evaluator.eval(dataloader)


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(get_free_device().index)

    baseline_composer = EmbeddedComposer([
        EmptyFileFilter(),
        InclusiveFileExtensionFilter(['.py']),
        FileGrainedChunker(),
        NegativePathDistanceRanker(),
        LexicographicSorter(),
        PathCommentHarvester(chunks_sep='\n\n', path_comment_template='# {filename}\n{content}'),
    ],
        pre_context_prompt='# {}\n',
        post_context_prompt='\n\n',
        path_comment_template='# {filename}\n{content}',
        recalculate_random_category=True,
    )

    config = OmegaConf.load('extra/cache/template.yaml')

    checkpointer_config = CheckpointManagerConfig(
        init_from=LoadingMode.BEST,
        main_metric='cross_entropy',
        directory=...)
    checkpointer = CheckpointManager(**checkpointer_config.dict)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        trust_remote_code=True,
    )

    for run_name in RUNS:
        run_dir = os.path.join(RUNS_DIR, run_name)
        checkpointer.directory = os.path.join(run_dir, 'checkpoints')
        run_script_file = os.path.join(run_dir, 'run_script.sh')

        composer_name = 'chained_composer/standard'
        with open(run_script_file) as stream:
            for line in stream.readlines():
                if line.startswith('composer='):
                    composer_name = line[9:].replace('\\', '').rstrip()
                    break

        composer_config_path = os.path.join(CONFIGS_DIR, 'composer', f'{composer_name}.yaml')
        composer_config = OmegaConf.load(composer_config_path)

        composer = init_composer(
            cls_name='chained_composer',
            loaded_config=composer_config,
            configs_dir=CONFIGS_DIR,
            tokenizer=tokenizer)
        composer.__class__ = EmbeddedComposer

        # initial model + native composer (1)
        config.output.results_filename = f'{run_name}_1.jsonl'
        config.model.model_name = MODEL_NAME
        run_eval_plcc(config, composer)

        # fine-tuned model + native composer (2)
        config.output.results_filename = f'{run_name}_2.jsonl'
        config.model.model_name = checkpointer.get_model_subdirectory()
        run_eval_plcc(config, composer)

        # fine-tuned model + baseline composer (3)
        config.output.results_filename = f'{run_name}_3.jsonl'
        config.model.model_name = checkpointer.get_model_subdirectory()
        run_eval_plcc(config, baseline_composer)


if __name__ == '__main__':
    main()
