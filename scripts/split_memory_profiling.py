from pipeline.configs.model_config import ModelConfig
from pipeline.model.adapters.split_adapter import SplitAdapter
from pipeline.model.init import init_model

import itertools

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def simulate_training_step(model: nn.Module,
                           optimizer: torch.optim.AdamW,
                           block_size: int,
                           num_blocks: int,
                           accumulation_steps: int,
                           ) -> None:
    input_ids = torch.randint(0, 10_000, (num_blocks, block_size), device=model.device)
    attention_mask = None

    for _ in range(accumulation_steps):
        model_output = model(input_ids, attention_mask)
        loss = F.cross_entropy(model_output.logits.flatten(0, 1), input_ids.flatten(0, 1))
        loss.backward()

    optimizer.step()


def estimate_grid(model: nn.Module,
                  adapter: SplitAdapter,
                  block_sizes: list[int],
                  num_blocks: list[int],
                  accumulation_steps: int,
                  ) -> torch.Tensor:
    optimizer = torch.optim.AdamW(
        params=adapter.get_trainable_parameters(model),
        fused=(model.device.type == 'cuda'))
    total_memory = torch.cuda.get_device_properties(model.device).total_memory

    oom_points = list()
    flat_grid = list()

    for point in tqdm(itertools.product(block_sizes, num_blocks), total=len(block_sizes) * len(num_blocks)):
        if any(point[0] >= p[0] and point[1] >= p[1] for p in oom_points):
            flat_grid.append(total_memory)
            continue

        try:
            simulate_training_step(model, optimizer, *point, accumulation_steps)
            torch.cuda.synchronize(model.device)
            flat_grid.append(torch.cuda.max_memory_allocated(model.device))
        except torch.cuda.OutOfMemoryError:
            flat_grid.append(total_memory)
            oom_points.append(point)

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(model.device)

    return torch.tensor(flat_grid).view(len(block_sizes), len(num_blocks)) / 2 ** 30


def plot_grid(grid: torch.tensor,
              block_sizes: torch.Tensor,
              num_blocks: torch.Tensor,
              num_gen_layers: int,
              device_name: str,
              ) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 8))

    contour_filled = ax.contourf(
        *torch.meshgrid(block_sizes, num_blocks, indexing='ij'),
        grid,
        cmap='inferno',
        levels=30)
    fig.colorbar(contour_filled, label='GB', ax=ax)

    ax.set_title(f'Maximum memory utilization : Split {num_gen_layers} : {device_name}')
    ax.set_xlabel('Block size [tokens]')
    ax.set_ylabel('Number of blocks')

    ax.set_xticks(block_sizes[(block_sizes == block_sizes.min()) | (block_sizes >= 512)])
    ax.set_yticks(num_blocks)

    lines_kwargs = dict(colors='white', linestyles='--', linewidth=1, alpha=0.1)
    ax.vlines(block_sizes, ymin=min(num_blocks), ymax=max(num_blocks), **lines_kwargs)
    ax.hlines(num_blocks, xmin=min(block_sizes), xmax=max(block_sizes), **lines_kwargs)

    return fig, ax


def main() -> None:
    # init parameters
    random_seed = 1337
    model_name = 'deepseek-ai/deepseek-coder-1.3b-base'
    cache_file = 'extra/cache/mem_grid_6.pt'
    plot_file = 'extra/viz/mem_grid_6.png'
    num_gen_layers = 6
    block_sizes = [2 ** power for power in range(6, 10 + 1)] + list(range(2 ** 11, 2 ** 13 + 1, 1024))
    num_blocks = list(range(1, 48 + 1))
    accumulation_steps = 2

    # set environment
    torch.manual_seed(random_seed)
    torch.set_float32_matmul_precision('high')

    # init model
    model_config = ModelConfig(
        tokenizer_name=model_name,
        model_name=model_name,
        trust_remote_code=True,
        load_from=None,
        compile=False)
    model = init_model(model_config)
    adapter = SplitAdapter(
        num_gen_layers=num_gen_layers,
        max_seq_len=max(num_blocks) * max(block_sizes) + 42,
        model_name=model_name,
        params_pattern='^generator.*')
    model = adapter.adapt(model)
    model = model.train().requires_grad_(True)

    grid = estimate_grid(model, adapter, block_sizes, num_blocks, accumulation_steps)
    block_sizes = torch.tensor(block_sizes)
    num_blocks = torch.tensor(num_blocks)

    torch.save({
        'grid': grid,
        'block_sizes': block_sizes,
        'num_blocks': num_blocks,
    }, cache_file)

    device_name = torch.cuda.get_device_properties(model.device).name
    fig, _ = plot_grid(grid, block_sizes, num_blocks, num_gen_layers, device_name)
    fig.savefig(plot_file)


if __name__ == '__main__':
    main()
