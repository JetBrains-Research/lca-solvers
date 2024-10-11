from pipeline.configs.checkpointer_config import CheckpointManagerConfig
from pipeline.configs.model_config import ModelConfig
from pipeline.data.composers.blocks.chunk_ranking import *
from pipeline.data.composers.init import init_composer
from pipeline.data.dataset import train_test_split, set_transform
from pipeline.data.preprocessors.init import init_preprocessor
from pipeline.environment.hardware import get_free_device
from pipeline.model.init import init_tokenizer
from pipeline.model.adapters.init import init_adapter
from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.checkpointers.data_structures import LoadingMode
from pipeline.trainers.validator import Validator

import json

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import AutoModelForCausalLM


MODEL_NAME = 'deepseek-ai/deepseek-coder-1.3b-base'
RUNS = [
    'encoder_fr_14_generator_tr_10_HP001',
    'encoder_fr_18_generator_tr_6_HP001',
]
ADAPTERS = [
    'configs/adapter/split_adapter/generator_10.yaml',
    'configs/adapter/split_adapter/generator_6.yaml',
]
RUNS_DIR = 'runs'
OUTPUTS_DIR = 'extra/outputs/eval_recalc'


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
        'random_cross_entropy',
        'inproject_top_3_accuracy',
        'inproject_top_5_accuracy',
        'inproject_top_10_accuracy'])
    checkpointer.init_from = LoadingMode.BEST

    model_config = OmegaConf.load('configs/model/dseek1p3.yaml')
    dataset_config = OmegaConf.load('configs/dataset/train_A100_server.yaml')
    split_config = OmegaConf.load('configs/split/256_5.yaml')
    # preprocessor_config = OmegaConf.load('configs/preprocessor/split_lm_preprocessor/full_input_loss_20k_40k.yaml')
    preprocessor_config = OmegaConf.load('configs/preprocessor/split_lm_preprocessor/full_input_loss_8k_16k.yaml')
    composer_config = OmegaConf.load('configs/composer/split_composer/python_files_20k_32.yaml')

    model_config = ModelConfig.from_dict(dict(model_config) | dict(load_from=None))
    tokenizer = init_tokenizer(model_config)

    composer = init_composer(
        cls_name='split_composer',
        loaded_config=composer_config,
        configs_dir='configs',
        tokenizer=tokenizer,
    )
    preprocessor = init_preprocessor(
        cls_name='split_lm_preprocessor1',
        loaded_config=preprocessor_config,
        tokenizer=tokenizer,
    )

    dataset = load_dataset(**dict(dataset_config))
    train_ds, valid_ds = train_test_split(dataset, **dict(split_config))
    set_transform(train_ds, valid_ds, composer, preprocessor)

    device = get_free_device()
    dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'

    for run_name, adapter_path in zip(RUNS, ADAPTERS):
        run_dir = os.path.join(RUNS_DIR, run_name)
        checkpointer.directory = os.path.join(run_dir, 'checkpoints')

        adapter_config = OmegaConf.load(adapter_path)
        adapter = init_adapter(
            cls_name='split_adapter',
            loaded_config=adapter_config,
            model_name=MODEL_NAME,
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            use_cache=False)
        model = adapter.adapt(model)

        generator_state_dict = model.generator.state_dict()
        safetensors_file = os.path.join(checkpointer.get_model_subdirectory(), 'generator/model.safetensors')
        with safe_open(safetensors_file, framework='pt', device=device.index) as stream:
            for k in stream.keys():
                generator_state_dict[k] = stream.get_tensor(k)

        model.generator.load_state_dict(generator_state_dict)
        model = model.eval().requires_grad_(False)

        validator = Validator(
            model=model,
            adapter=adapter,
            valid_metrics=valid_metrics,
            valid_ds=valid_ds,
            batch_size=1,
            num_workers=8,
            prefetch_factor=8,
        )

        validation_results = {
            'checkpoint': checkpointer.get_model_subdirectory(),
        } | validator.validate()
        with open(os.path.join(OUTPUTS_DIR, f'{run_name}.json'), 'w') as stream:
            json.dump(validation_results, stream)


if __name__ == '__main__':
    main()
