import subprocess
import warnings

import torch


def get_free_device(used_memory_upper_bound: float = 0.001) -> torch.device:
    if hasattr(get_free_device, 'allocated'):
        return get_free_device.allocated

    for gpu_index in range(torch.cuda.device_count()):
        gpu_pid_stats = subprocess.check_output([
            'nvidia-smi', f'-i={gpu_index}', '--query-compute-apps=pid', '--format=csv,noheader',
        ], encoding='utf-8')
        gpu_mem_stats = subprocess.check_output([
            'nvidia-smi', f'-i={gpu_index}', '--query-gpu=memory.used,memory.total', '--format=csv,noheader',
        ], encoding='utf-8')

        mem_used, mem_total = map(int, gpu_mem_stats.replace('MiB', '').split(', '))

        if not gpu_pid_stats and mem_used / mem_total <= used_memory_upper_bound:
            get_free_device.allocated = torch.device(f'cuda:{gpu_index}')
            return get_free_device.allocated

    warnings.warn('No CUDA devices were found. CPU will be used.')
    return torch.device('cpu')


def get_optimal_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    else:
        warnings.warn('torch.bfloat16 is not supported. torch.float16 '
                      'with gradient scaling will be used instead.')
        return torch.float16
