from pipeline.model.adapters.adapter_base import AdapterBase

from typing import Any

import torch
import torch.nn as nn


class PrefixUnmaskAdapter(AdapterBase):
    def get_args_kwargs(self,
                        input_ids: torch.Tensor,
                        target_ids: torch.Tensor,
                        loss_mask: torch.Tensor,
                        completion_mask: torch.Tensor,
                        category_ids: torch.Tensor,
                        input_attn_mask: torch.Tensor,
                        target_attn_mask: torch.Tensor,
                        ) -> tuple[tuple[Any], dict[str, Any]]:
        pass  # TODO

    def adapt(self, model: nn.Module) -> nn.Module:
        pass  # TODO
