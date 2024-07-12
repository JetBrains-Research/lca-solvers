import click
from array import array
import wandb
from huggingface_hub import notebook_login
import sys
import gc
import math
from typing import List

import numpy as np
import random
from collections import Counter
import os
import deepspeed

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator, DeepSpeedPlugin

from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

sys.path.append('/home/blatova/lca-solvers/')

from data_filters.repo_snapshot_filter_stack import SnapshotFilterStack
from data_classes.datapoint_composed import DatapointComposed
from context_composers.context_composer_path_distance import ContextComposerPathDistance

from huggingface_hub import HfApi, HfFolder, Repository, RepoCard

from finetune.ploting import live_plot
from finetune.composing_context import compose_input_sequence
from finetune.working_with_data import get_new_val_dataset, is_validation, is_train, get_new_ema, get_standard_val_dataset, get_train_and_val_loader
from finetune.train_and_validate import make_validation_step, train

ACCUM_STEPS_NUM = 64
VALIDATION_PERIOD = 10
EMA_PERIOD = 40
EMA_ALPHA = 2/(EMA_PERIOD+1)

os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"

def parse_devices(ctx, param, value):
    try:
        return [int(x) for x in value.split(',')]
    except ValueError:
        raise click.BadParameter('Devices should be a comma-separated list of integers.')


def init(adam_lr, DEVICES_TO_TRAIN, parallel_with_accelerate, context_max_len_tokens):
    wandb.login()
    notebook_login()
    wandb_config={
        "ACCUM_STEPS_NUM": ACCUM_STEPS_NUM,
        "VALIDATION_PERIOD": VALIDATION_PERIOD,
        "CONTEXT_MAX_LEN_TOKENS": context_max_len_tokens,
        "EMA_PERIOD": EMA_PERIOD, 
        "VAL_DATASET_SIZE": 128,
        "ADAMW_LR": adam_lr, 
        "ADAMW_weight_decay":0.01,
        "Cosine_T_max":50, 
        "Cosine_eta_min":1e-6,
        # "num_GPU_cards": 2,
        "DEVICES_TO_TRAIN":DEVICES_TO_TRAIN,
        "accelerate": parallel_with_accelerate,
        "DeepSpeed_plugin": False
    }

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="finetuning_with_lora",
    #     # track hyperparameters and run metadata
    #     config=wandb_config
    # )



def get_model_for_training(use_lora=False):
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", 
                                                 trust_remote_code=True, torch_dtype=torch.bfloat16, device_map = "auto", attn_implementation="flash_attention_2")
    if use_lora:
        peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1, target_modules='all-linear'
)
        model_lora = get_peft_model(model, peft_config)
        return model
    else:
        return model
    


@click.command()
@click.option('--use_lora', default=False)
@click.option('--devices_to_train', default='6', callback=parse_devices, help='Comma-separated list of devices to train on.')
@click.option('--adam_lr', default=5e-5)
@click.option('--std_val_dataset_version', default="v0")
@click.option('--save_standard_dataset', default=False)
@click.option('--use_standard_dataset', default=True)
@click.option('--suffix_to_save_on_huggingfaces', default="")
@click.option('--parallel_with_accelerate', default = False)
@click.option('--context_max_len_tokens', default=2000)
def main(use_lora: bool, 
         devices_to_train, #: List[int]
        adam_lr: float,
        std_val_dataset_version: str,
        save_standard_dataset: bool,
        use_standard_dataset: bool,
         suffix_to_save_on_huggingfaces: str,
         parallel_with_accelerate: bool,
         context_max_len_tokens: int
        ):
    init(adam_lr, devices_to_train, parallel_with_accelerate, context_max_len_tokens)
    devices_to_train_str = ','.join(str(x) for x in devices_to_train)
    composer = ContextComposerPathDistance()
    model = get_model_for_training(use_lora)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=adam_lr, weight_decay=0.01) #added weight decay
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    train_loader, val_loader = get_train_and_val_loader(use_standard_dataset, save_standard_dataset, std_val_dataset_version) 
    if parallel_with_accelerate:
        accelerator = Accelerator(split_batches=False, mixed_precision="no", gradient_accumulation_steps=ACCUM_STEPS_NUM)
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # TODO
    train()

if __name__ == "__main__":
    main()
        
        
        
        
        
        
    




