import torch
from torch import Tensor
from typing import Optional


def loss(
        predict: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        feature_dim: Optional[int] = None,
        event_weight: Optional[Tensor] = None
):
    if event_weight is not None:
        if mask is not None:
            den = torch.sum(mask.float() * event_weight.unsqueeze(-1)) * feature_dim
            if den == 0:
                return torch.tensor(0.0, device=predict.device, dtype=predict.dtype)
            return torch.sum(((predict - target) ** 2) * mask.float() * event_weight.unsqueeze(-1)) / den
        else:
            return torch.sum(((predict - target) ** 2) * event_weight.unsqueeze(-1)) / torch.sum(event_weight)
    else:
        if mask is not None:
            den = torch.sum(mask.float()) * feature_dim
            if den == 0:
                return torch.tensor(0.0, device=predict.device, dtype=predict.dtype)
            return torch.sum(((predict - target) ** 2) * mask.float()) / den
        else:
            return torch.mean((predict - target) ** 2)
