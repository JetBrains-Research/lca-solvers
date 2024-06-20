import click
import jsonlines
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftModelForCausalLM
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_classes.datapoint_composed import DatapointComposed

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)

PEFT_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1,
    target_modules='all-linear'
)


def compose_input_sequence(dp: DatapointComposed, max_seq_len: int):
    context = dp.context[0]
    completion = dp.completion[0]
    seq = context + completion
    while len(seq) < max_seq_len:
        seq += completion

    return seq[-max_seq_len:]


class HFDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset['train']

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]


def get_data_loader() -> DataLoader:
    dataset = load_dataset("JetBrains-Research/context-py-train", "path_distance_relevant",
                           cache_dir='/mnt/data2/shared-data/lca/hf_cache/')
    torch_dataset = HFDataset(dataset)
    loader = DataLoader(torch_dataset, batch_size=1, shuffle=True)
    return loader



def get_lora_model(torch_dtype: torch.dtype = torch.bfloat16,
                   lora_config: LoraConfig = PEFT_CONFIG,
                   device: torch.device = torch.device("cuda:2"),
                   enable_lora: bool = True,
                   ) -> PeftModelForCausalLM | AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation='flash_attention_2',
    ).to(device)
    print('Model Ready')

    if enable_lora:
        model_lora = get_peft_model(model, lora_config)
        print('Lora Ready')

        return model_lora

    return model

def get_memory(device: torch.device):
    memory_consumption = (torch.cuda.max_memory_allocated(device))
    torch.cuda.reset_max_memory_allocated(device)
    return memory_consumption

@click.command()
@click.option('--max-seq-len', default=512)
@click.option('--enable-lora', default=True)
@click.option('--zero-grad-none', default=False)
def main(
        max_seq_len: int,
        enable_lora: bool,
        zero_grad_none: bool,
):
    # torch.cuda.memory._record_memory_history(max_entries=100000)

    loader = get_data_loader()
    model_lora = get_lora_model(enable_lora=enable_lora)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model_lora.parameters(), lr=5e-5)
    memory_list = list()
    memory_list.append(get_memory(model_lora.device))
    print(memory_list[0])
    grad_accum = 16
    in_token_num = 0
    optimizer.zero_grad()
    is_oom = False
    for idx, hf_dp in enumerate(loader):
        try:
            torch.cuda.empty_cache()
            mem_res = dict()

            dp = DatapointComposed.from_hf_datapoint(hf_dp)
            input_seq = compose_input_sequence(dp, max_seq_len * 10)
            inputs = tokenizer(input_seq, return_tensors='pt', truncation=True, max_length=max_seq_len).to(model_lora.device)
            # print(inputs["input_ids"].shape)
            # outputs = model_lora(labels=inputs["input_ids"], **inputs)
            outputs = model_lora(**inputs)
            in_token_num += inputs["input_ids"].numel()
            mem_res['in_tokens'] = in_token_num
            logits_size = outputs['logits'].size(-1)
            completion_len = int(max_seq_len/4)
            loss_completion = criterion(outputs['logits'].view(-1, outputs['logits'].size(-1))[-completion_len:-1, :],
                                        inputs['input_ids'].view(-1)[-completion_len + 1:])
            loss_completion.backward()
            # (outputs.loss/grad_accum).backward()
            if (idx + 1) % grad_accum == 0:
                for param in model_lora.parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            param.grad /= grad_accum
                optimizer.step()
                optimizer.zero_grad(set_to_none=zero_grad_none)  # (set_to_none=True)
                in_token_num = 0
            mem_res['max_memory'] = get_memory(model_lora.device)
            memory_list.append(mem_res)
            print(f'{mem_res["in_tokens"] / 1e3:.3f}K tokens: {mem_res["max_memory"]/1e9 :.3f}GB', end=' | ')
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            is_oom = True

        if idx > 100:
            # torch.cuda.memory._dump_snapshot(f"pt_mem_snapshot_{max_seq_len}.pickle")
            # torch.cuda.memory._record_memory_history(enabled=None)
            if len(memory_list) > 1:
                max_memory = max([mem_res["max_memory"] for mem_res in memory_list[1:]])
            else:
                max_memory = None
            experiment_result = {
                'model_memory': memory_list[0],
                'max_seq_len': max_seq_len,
                'enable_lora': enable_lora,
                'max_memory': max_memory,
                'set_to_none_zero_grad': zero_grad_none,
                'OOM': is_oom,
            }
            with jsonlines.open('memory_consumption_results.jsonl', 'a') as writer:
                writer.write(experiment_result)

            print()
            return True

    return False


if __name__ == '__main__':
    main()
