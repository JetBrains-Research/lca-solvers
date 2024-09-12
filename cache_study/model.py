import copy

import torch.nn as nn


def create_model_reference(model: nn.Module) -> nn.Module:
    model_ref = copy.deepcopy(model)
    for orig_param, ref_param in zip(model.parameters(), model_ref.parameters()):
        ref_param.data = orig_param.data
    return model_ref


def split_model(model: nn.Module, num_gen_blocks: int) -> tuple[nn.Module, nn.Module]:
    indexer = create_model_reference(model.model)
    indexer.layers = indexer.layers[:-num_gen_blocks]
    indexer.norm = nn.Identity()

    generator = create_model_reference(model)
    generator.model.layers = generator.model.layers[-num_gen_blocks:]

    return indexer, generator
