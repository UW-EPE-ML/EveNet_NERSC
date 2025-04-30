import torch
from torch import Tensor
from typing import Optional


def loss(
        predict: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        feature_dim: Optional[int] = None,
):
    if mask is not None:
        return torch.sum(((predict - target) ** 2) * mask.float()) / (torch.sum(mask.float()) * feature_dim)
    else:
        return torch.mean((predict - target) ** 2)
