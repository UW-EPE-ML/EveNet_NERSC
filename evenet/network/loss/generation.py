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
        den = torch.sum(mask.float()) * feature_dim
        if den == 0:
            return torch.tensor(0.0, device=predict.device, dtype=predict.dtype)
        return torch.sum(((predict - target) ** 2) * mask.float()) / den
    else:
        return torch.mean((predict - target) ** 2)
