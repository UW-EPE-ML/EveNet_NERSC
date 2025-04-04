import torch
import torch.nn.functional as F

def loss(predict, target, class_weight=None, reduction='none'):
    """
    Cross-entropy loss with optional class weights.
    Returns per-sample loss for external weighting.

    Args:
        predict (Tensor): (N, C) logits
        target (Tensor): (N,) class indices
        class_weight (Tensor, optional): (C,) class weights
        reduction (str): 'none', 'mean', or 'sum'

    Returns:
        Tensor: (N,) if reduction='none', else scalar
    """
    return F.cross_entropy(
        input=predict,
        target=target,
        weight=class_weight,
        reduction=reduction
    )