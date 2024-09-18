import re
from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch
import torch.nn as nn
from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers import AutoConfig, LlamaForCausalLM


class AdapterBase(ABC):
    def __init__(self, model_name: str, params_pattern: str | None) -> None:
        self.model_name = model_name
        self.params_pattern = params_pattern

        hf_model_config = AutoConfig.from_pretrained(model_name)
        model_cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(hf_model_config)]
        assert model_cls == LlamaForCausalLM, 'Different architectures may require individual adaptations.'

    def get_trainable_parameters(self, model: nn.Module) -> Iterable[torch.Tensor]:
        return [
            params for name, params in model.named_parameters()
            if re.search(self.params_pattern, name)
        ] if self.params_pattern is not None else model.parameters()

    @abstractmethod
    def get_args_kwargs(self,
                        input_ids: torch.Tensor,
                        target_ids: torch.Tensor,
                        loss_mask: torch.Tensor,
                        completion_mask: torch.Tensor,
                        category_ids: torch.Tensor,
                        input_attn_mask: torch.Tensor,
                        target_attn_mask: torch.Tensor,
                        ) -> tuple[tuple[Any], dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def adapt(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError
