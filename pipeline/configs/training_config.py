import subprocess
import warnings
from dataclasses import dataclass

import torch


def get_free_device(used_memory_upper_bound: float = 0.001) -> torch.device:
    for gpu_index in range(torch.cuda.device_count()):
        gpu_pid_stats = subprocess.check_output([
            'nvidia-smi', f'-i={gpu_index}', '--query-compute-apps=pid', '--format=csv,noheader',
        ], encoding='utf-8')
        gpu_mem_stats = subprocess.check_output([
            'nvidia-smi', f'-i={gpu_index}', '--query-gpu=memory.used,memory.total', '--format=csv,noheader',
        ], encoding='utf-8')

        mem_used, mem_total = map(int, gpu_mem_stats.replace('MiB', '').split(', '))

        if not gpu_pid_stats and mem_used / mem_total <= used_memory_upper_bound:
            return torch.device(f'cuda:{gpu_index}')

    warnings.warn('No CUDA devices were found. The CPU will be used.')
    return torch.device('cpu')


def get_optimal_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    else:
        return torch.float16


@dataclass
class TrainingConfig:  # TODO: default values
    # Iteration parameters
    max_iters: int
    valid_freq: int
    gradient_accumulation_steps: int
    micro_batch_size: int

    # AdamW optimizer
    learning_rate: float
    beta_1: float
    beta_2: float
    weight_decay: float
    max_grad_norm: float

    # Cosine lr scheduler with warmup
    decay_lr: bool
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float

    # DataLoader
    shuffle: bool
    drop_last: bool
    num_workers: int
    random_seed: int

    # Hardware
    device: torch.device = get_free_device()
    dtype: torch.dtype = get_optimal_dtype()
    compile: bool = True
