import torch


def crop_tail_inplace(tensor: torch.Tensor, length: int) -> None:
    if length >= tensor.shape[-1]:
        return

    tail = tensor[..., -length:].clone()
    tensor.resize_(tail.shape)
    tensor.copy_(tail)
    assert torch.equal(tensor, tail)
