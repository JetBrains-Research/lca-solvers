import random
import re

import click
import torch
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedTokenizerBase, BatchEncoding, PreTrainedModel


class SplitIndexerBase:
    def __init__(self, list_split_points: list[int] | None = None):
        self.list_split_points = list_split_points

    def is_list_correct(self, decoder_layers) -> bool:
        if len(self.list_split_points) % 2 != 0:
            return False
        for idx in self.list_split_points:
            if idx not in range(len(decoder_layers)):
                return False
        return True

    def check_idx(self, idx: int) -> bool:
        return idx in self.list_split_points


class HiddenStatesSplitter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device_num: int, split_points_strategy, split_size: int = 4):
        self.tokenizer = tokenizer
        self.split_size = split_size
        self.device = torch.device(device_num) if torch.cuda.is_available() else torch.device('cpu')
        self._hook_handles: list[RemovableHandle] | None = None
        self.split_points_strategy = split_points_strategy

    def _get_decoder_layers(self, model: PreTrainedModel):
        if hasattr(model, 'model'):
            if hasattr(model.model, 'layers'):
                decoder_layers = model.model.layers
            else:
                raise NotImplementedError('Cannot access decoder layers')
        else:
            raise NotImplementedError('Cannot access decoder layers')
        return decoder_layers

    def _strategy_parser(self) -> tuple[str, int] | tuple[str, float] | tuple[str, int, int]:
        first_match = re.match(r"^first_(\d+)$", self.split_points_strategy)
        if first_match:
            return 'first', int(first_match.group(1))
        last_match = re.match(r"^last_(\d+)$", self.split_points_strategy)
        if last_match:
            return 'last', int(last_match.group(1))
        mid_match = re.match(r"^middle_(\d+)_(\d+)$", self.split_points_strategy)
        if mid_match:
            return 'middle', int(mid_match.group(1)), int(mid_match.group(2))
        random_match = re.match(r"^random_(\d*\.\d+)$", self.split_points_strategy)
        if random_match:
            return 'random', float(random_match.group(1))
        raise NotImplementedError(f'strategy `{self.split_points_strategy}` is not supported')

    def _get_split_points(self, decoder_layers) -> list[int]:
        parsed_strategy = self._strategy_parser()
        if parsed_strategy[0] == 'first':
            return [0, parsed_strategy[1]]
        elif parsed_strategy[0] == 'last':
            return [parsed_strategy[1], len(decoder_layers) - 1]
        elif parsed_strategy[0] == 'middle':
            return sorted([parsed_strategy[1], parsed_strategy[2]])
        elif parsed_strategy[0] == 'random':
            split_points = list()
            for idx, _ in enumerate(decoder_layers):
                if random.random() < parsed_strategy[1]:
                    split_points.append(idx)

            if len(split_points) % 2 == 0:
                return split_points

            if len(decoder_layers) - 1 in split_points:
                return split_points[:-1]
            else:
                split_points.append(len(decoder_layers) - 1)
                return split_points
        else:
            raise NotImplementedError(f'There is no splitting algorythm for {parsed_strategy[0]} strategy')

    def add_hooks(self, model: PreTrainedModel) -> None:
        split_is_on = False
        if self._hook_handles is None:
            self._hook_handles = list()

        layers = self._get_decoder_layers(model)
        split_points = self._get_split_points(layers)
        split_checker = SplitIndexerBase(split_points)
        if not split_checker.is_list_correct(layers):
            raise ValueError('split_points is not in the right format')

        for idx, decoder_layer in enumerate(layers):
            if split_is_on:
                hook_handle = decoder_layer.register_forward_pre_hook(self.kwarg_adjuster_pre_hook, with_kwargs=True)
                self._hook_handles.append(hook_handle)

            if idx in split_points and not split_is_on:
                hook_handle = decoder_layer.register_forward_hook(self.split_layer_hook_fn)
                split_is_on = True
                self._hook_handles.append(hook_handle)
            elif idx in split_points and split_is_on:
                hook_handle = decoder_layer.register_forward_hook(self.compose_layer_hook_fn)
                split_is_on = False
                self._hook_handles.append(hook_handle)

    def remove_hooks(self) -> None:
        if self._hook_handles is not None:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles = None

    def tokenize_for_splitting(self, texts: list[str]) -> BatchEncoding:
        """
        This method performs tokenization that returns a tokenized sequence of length divisible by `split_size`.
        Splitter may work inconsistent with regular tokenization.
        :param texts: list of texts to tokenize
        :return: tokenized output that's on the device
        """
        split_size = self.split_size
        tokenized_prompt = self.tokenizer(texts, return_tensors='pt', padding=True)

        seq_len = tokenized_prompt['input_ids'].size(-1)

        if seq_len % split_size != 0:
            pad_size = (split_size - seq_len % split_size) % split_size
            tokenized_prompt = self.tokenizer(texts, return_tensors="pt", padding='max_length',
                                         max_length=seq_len + pad_size)

        tokenized_prompt.to(self.device)
        return tokenized_prompt

    def split_layer_hook_fn(self, module, module_input, module_output):
        hs = module_output[0]
        batch_size, seq_len, hidden_dim = hs.size()

        split_size = self.split_size
        pad_size = (split_size - seq_len % split_size) % split_size

        # Padding the hidden states to make the sequence length divisible by split_size
        padded_hs = F.pad(hs, (0, 0, pad_size, 0), mode="constant", value=0)

        if not padded_hs.is_contiguous():
            padded_hs = padded_hs.contiguous()
        new_batch_size = batch_size * split_size
        new_seq_len = (seq_len + pad_size) // split_size
        splitted_hs = padded_hs.view(new_batch_size, new_seq_len, hidden_dim)

        # print(splitted_hs.shape)

        return (splitted_hs,) + module_output[1:]

    def compose_layer_hook_fn(self, module, module_input, module_output):
        splitted_hs = module_output[0]
        split_size = self.split_size

        batch_size = splitted_hs.size(0) // split_size
        seq_len = splitted_hs.size(1) * split_size
        hidden_dim = splitted_hs.size(2)

        if not splitted_hs.is_contiguous():
            splitted_hs = splitted_hs.contiguous()
        composed_hs = splitted_hs.view(batch_size, seq_len, hidden_dim)

        return (composed_hs,) + module_output[1:]

    def kwarg_adjuster_pre_hook(self, module, module_input, kwargs):
        split_size = self.split_size

        def _split_position_tensor(tensor, _split_size=split_size):
            seq_len = tensor.size(1)
            pad_size = (_split_size - (seq_len % _split_size)) % _split_size

            padded_tensor = F.pad(tensor, (pad_size, 0), mode="constant", value=0)

            if not padded_tensor.is_contiguous():
                padded_tensor = padded_tensor.contiguous()
            return padded_tensor.view(padded_tensor.size(0) * _split_size,
                                      padded_tensor.size(1) // _split_size)

        kwargs['position_ids'] = _split_position_tensor(kwargs['position_ids'])
        kwargs['position_embeddings'] = None

        # Handle 'attention_mask' based on the attention implementation
        if kwargs['attention_mask'] is not None:
            if 'flash_attention' in module.self_attn.config._attn_implementation:
                # flash_attention mask is of size B x L
                kwargs['attention_mask'] = _split_position_tensor(kwargs['attention_mask'])
            else:
                raise NotImplementedError("Only flash_attention is supported at this time.")

        return module_input, kwargs


@click.command()
@click.option('--strategy', '-s')
@click.option('--device-num', '-d', type=int, default=0)
def splitter(strategy: str, device_num):
    device = torch.device(device_num) if torch.cuda.is_available() else torch.device("cpu")
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.truncation_side = "left"

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )

    hs_splitter = HiddenStatesSplitter(tokenizer=tokenizer, device_num=device_num, split_points_strategy=strategy)
    prompt = 'def add(a: int, b: int) -> int:\n\treturn a + b\n\n'
    tokenized_input = hs_splitter.tokenize_for_splitting([prompt])

    print(hs_splitter._strategy_parser())
    print()
    decoder_layers = hs_splitter._get_decoder_layers(model)
    print(hs_splitter._get_split_points(decoder_layers))

    out = model(**tokenized_input, output_hidden_states=True)
    print([hs.shape for hs in out.hidden_states])

    print()
    hs_splitter.add_hooks(model)
    out = model(**tokenized_input, output_hidden_states=True)
    print([hs.shape for hs in out.hidden_states])

    print()
    hs_splitter.remove_hooks()
    out = model(**tokenized_input, output_hidden_states=True)
    print([hs.shape for hs in out.hidden_states])

    print()
    hs_splitter.add_hooks(model)
    out = model(**tokenized_input, output_hidden_states=True)
    print([hs.shape for hs in out.hidden_states])

    print()
    hs_splitter.remove_hooks()
    out = model(**tokenized_input, output_hidden_states=True)
    print([hs.shape for hs in out.hidden_states])


if __name__ == '__main__':
    splitter()