import torch
import torch.nn.functional as F


def loss(predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, beta: float = 1.0) -> torch.Tensor:
    """
    Compute Smooth L1 (Huber) regression loss per event.

    Args:
        predict (torch.Tensor): Predicted values, shape (N, D)
        target (torch.Tensor): Ground truth values, shape (N, D)
        mask (torch.Tensor, optional): Optional mask to ignore certain dimensions, shape (N, D). If None, all dimensions are considered.
        beta (float): Transition point from L1 to L2 loss.

    Returns:
        torch.Tensor: Per-event loss, shape (N,)
    """
    # Compute element-wise Smooth L1 loss
    loss_per_dim = F.smooth_l1_loss(predict, target, reduction='none', beta=beta)  # shape: (N, D)

    if mask is not None:
        # Compute valid mask per event
        valid = mask.sum(dim=1) > 0

        if valid.any():
            # Only keep valid events
            masked_loss = loss_per_dim[valid] * mask[valid]

            # Normalize by number of valid features
            valid_counts = mask[valid].sum(dim=1).clamp(min=1.0)
            loss_per_event = masked_loss.sum(dim=1) / valid_counts
        else:
            # No valid regression targets in the entire batch
            return torch.tensor(0.0, device=predict.device)
    else:
        # No mask, just mean across features
        loss_per_event = loss_per_dim.mean(dim=1)

    return loss_per_event
