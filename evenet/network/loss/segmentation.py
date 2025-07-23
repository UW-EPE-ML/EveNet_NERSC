from collections import OrderedDict
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

def sigmoid_focal_loss(inputs, targets, mask = None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. (B, N, P)
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                shape: (B, N, P)
        mask: (optional) A boolean tensor of the shape: (B,N)
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor (B, N) where B is the batch size and N is the number of elements in each example.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (B, N, P)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss # (B, N, P)

    if mask is not None:
        loss = loss * mask.float().unsqueeze(-1)  # (B, N, P)

    return loss

def hungarian_matching(
    predict_cls: Tensor,
    predict_mask: Tensor,
    target_cls: Tensor,
    target_mask: Tensor,
    class_weight: Tensor = None,
    segmentation_mask: Tensor = None
):
    """
    Perform Hungarian matching between predicted and target masks.

    Args:
        predict_cls (Tensor): (B, N, C) logits for class predictions
        predict_mask (Tensor): (B, N, P) logits for mask predictions
        target_cls (Tensor): (B, N, C) class labels
        target_mask (Tensor): (B, N, P) mask labels
        class_weight (Tensor, optional): (C,) class weights
        segmentation_mask (Tensor, optional): (B, N) mask to ignore certain pixels

    Returns:
        Tuple[Tensor, Tensor]: matched indices and matched masks
    """
    # Placeholder for the actual implementation of Hungarian matching
    # This function should return the matched indices and masks based on the inputs.
    num_queries = predict_cls.shape[1]
    predict_mask_expanded = predict_mask.unsqueeze(2).repeat(1, 1, num_queries, 1)  # (B, N_pred, N_tgt, P)
    target_mask_expanded = target_mask.unsqueeze(1).repeat(1, num_queries, 1, 1)  # (B, N_pred, N_tgt, P)
    segmentation_mask_expanded = segmentation_mask.unsqueeze(1).repeat(1, num_queries, 1) if segmentation_mask is not None else None # (B, N_pred, N_tgt)

    mask_cost = sigmoid_focal_loss(
        inputs = predict_mask_expanded,
        targets = target_mask_expanded,
        mask = None,
        alpha = 1,
        gamma = 1
    ) # (B, N_pred, N_tgt, P)

    mask_cost = mask_cost.sum(dim=-1)  # Sum over the last dimension (P) to get (B, N, N)

    src_indices, tgt_indices = assignment_to_indices(batch_linear_assignment(mask_cost))

    return src_indices, tgt_indices




def loss(
        predict_cls,
        predict_mask,
        target_cls,
        target_mask,
        class_weight=None,
        segmentation_mask=None,
        reduction='none'
    ):
    """

    Args:
        predict_cls (Tensor): (B, N, C) logits for class predictions
        predict_mask (Tensor): (B, N, P) logits for mask predictions
        target_cls (Tensor): (B, N, C) class labels
        target_mask (Tensor): (B, N, P) mask labels
        class_weight (Tensor, optional): (C,) class weights
        segmentation_mask (Tensor, optional): (B, N) mask to ignore certain pixels
        reduction (str): 'none', 'mean', or 'sum'

    Returns:
        Tensor: (B,N) if reduction='none', else scalar
    """

    with torch.no_grad():
        pred_indices, tgt_indices = hungarian_matching(
            predict_cls=predict_cls,
            predict_mask=predict_mask,
            target_cls=target_cls,
            target_mask=target_mask,
            class_weight=class_weight,
            segmentation_mask=segmentation_mask
        )

    # Gather the predicted and target masks based on the matched indices
    predict_mask_best = predict_mask[pred_indices]  # (B, N, P)
    target_mask_best = target_mask[tgt_indices]  # (B, N, P)
    predict_class_best = predict_cls[pred_indices]  # (B, N, C)
    target_class_best = target_cls[tgt_indices]  # (B, N, C)
    target_mask_best = target_mask[tgt_indices]  # (B, N)




    return F.cross_entropy(
        input=predict,
        target=target,
        weight=class_weight,
        reduction=reduction,
        ignore_index=-1,
    )