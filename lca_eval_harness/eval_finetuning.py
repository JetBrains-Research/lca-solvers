import gc

import torch
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest

from lca_eval_harness.dataset_loaders.hf_data_loader import HFDataLoader
from lca_eval_harness.eval import Evaluator
from lca_eval_harness.model_inference.vllm_inference import VllmEngine

model_paths = [
    f'jenyag/model_lora_v{i}' for i in range(5)
]
base_model_path = 'deepseek-ai/deepseek-coder-1.3b-base'


def main():
    data_loader = HFDataLoader(hf_dataset_path='ekaterina-blatova-jb/val_dataset_lora')

    for model_v, model_path in enumerate(model_paths):
        # deepseek_ft_path = snapshot_download(repo_id=model_path)
        inference_engine = VllmEngine(
            hf_model_path=model_path,
            max_model_len=16_000,
            dtype=torch.bfloat16,
            # tensor_parallel_size=2,
        )
        sampling_params = {
            'temperature': 0.0,
            'min_tokens': 10,
        }
        evaluator = Evaluator(
            inference_engine=inference_engine,
            sampling_params=sampling_params,
            data_loader=data_loader
        )
        # evaluator.compose_dummy_prompts(prefix_max_len=10_000, prompt_max_len=12_000)
        evaluator.compose_random_prompts(prefix_max_len=2_000, prompt_max_len=8_000)
        evaluator.get_outputs()
        model_name = model_path.split('/')[-1]
        evaluator.save_results(
            filename=f'deepseek1b_finetuning/{model_name}_short_input.jsonl'
        )

        del inference_engine
        del evaluator

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
