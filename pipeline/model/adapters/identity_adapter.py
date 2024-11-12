from pipeline.data.preprocessors.preprocessor_base import BatchMetadata
from pipeline.model.adapters.adapter_base import AdapterBase

from typing import Any

import torch
import torch.nn as nn


class IdentityAdapter(AdapterBase):
    def get_args_kwargs(self,
                        input_ids: torch.Tensor,
                        _target_ids: torch.Tensor,
                        _loss_mask: torch.Tensor,
                        _completion_mask: torch.Tensor,
                        _category_ids: torch.Tensor,
                        input_attn_mask: torch.Tensor,
                        _target_attn_mask: torch.Tensor,
                        _metadata: BatchMetadata,
                        ) -> tuple[tuple[Any], dict[str, Any]]:
        args = (input_ids,)
        kwargs = dict(attention_mask=input_attn_mask)
        return args, kwargs

    def adapt(self, model: nn.Module) -> nn.Module:
        return model
