from cache_study.model import init_tokenizer_model
from cache_study.pipeline import BOSUsage, Pipeline
from composers.chain.blocks.chunk_ranking import NegativePathDistanceRanker
from composers.chain.blocks.chunk_sorting import LexicographicSorter
from composers.chain.blocks.file_chunking import FileGrainedChunker
from composers.chain.blocks.file_filtering import EmptyFileFilter, InclusiveFileExtensionFilter
from composers.chain.chain import UnsafeComposerChain

import os
from dataclasses import asdict

import hydra
import jsonlines
from datasets import load_dataset
from datasets.config import HF_DATASETS_CACHE
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm.auto import tqdm

OUTPUTS_DIR = 'outputs'
COMPOSER = UnsafeComposerChain(
    EmptyFileFilter(),
    InclusiveFileExtensionFilter(['.py']),
    FileGrainedChunker(),
    NegativePathDistanceRanker(),
    LexicographicSorter(),
)


@hydra.main(config_path='configs', version_base=None)
def main(config: DictConfig) -> None:
    eval_name = HydraConfig.get().job.config_name
    output_path = os.path.join(OUTPUTS_DIR, f'{eval_name}.jsonl')

    dataset = load_dataset(
        path='JetBrains-Research/lca-project-level-code-completion',
        name=config.dataset,
        split='test',
        cache_dir=HF_DATASETS_CACHE)
    tokenizer, model = init_tokenizer_model(config.model)

    pipeline = Pipeline(
        composer=COMPOSER,
        max_num_blocks=config.max_num_blocks,
        tokenizer=tokenizer,
        bos_usage=BOSUsage(config.bos),
        full_model=model,
        num_gen_layers=config.num_gen_layers,
    )

    for datapoint in tqdm(dataset):
        output = pipeline(datapoint)
        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(asdict(output))


if __name__ == '__main__':
    main()
