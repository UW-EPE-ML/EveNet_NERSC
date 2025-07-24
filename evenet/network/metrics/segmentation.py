from typing import Callable

import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch.nn.functional as F

from evenet.utilities.debug_tool import time_decorator


class SegmentationMetrics:
    def __init__(self):
        pass
    def update(self, y_true: torch.Tensor, y_pred_raw: torch.Tensor):
        pass
    def reset(self, cm: bool = True, logits: bool = True):
        pass
    def reduce_across_gpus(self):
        """All-reduce across DDP workers"""
        if torch.distributed.is_initialized():
            tensor = torch.tensor(self.matrix, dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            self.matrix = tensor.cpu().numpy()

            valid_tensor = torch.tensor([self.valid], dtype=torch.long, device=self.device)
            total_tensor = torch.tensor([self.total], dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(valid_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
            self.valid = valid_tensor.item()
            self.total = total_tensor.item()

            hist_store = torch.tensor(self.hist_store, dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(hist_store, op=torch.distributed.ReduceOp.SUM)
            self.hist_store = hist_store.cpu().numpy()

    def compute(self, matrix=None):
        pass

    def assign_train_result(self, train_hist_store=None, train_matrix=None):
        self.train_hist_store = train_hist_store
        self.train_matrix = train_matrix

@time_decorator(name="[Segmentation] shared_step")
def shared_step(
        target_classification: torch.Tensor,
        target_mask: torch.Tensor,
        predict_classification: torch.Tensor,
        predict_mask: torch.Tensor,
        segmentation_mask: torch.Tensor,
        point_cloud_mask: torch.Tensor,
        seg_loss_fn: Callable,
        class_weight: torch.Tensor,
        loss_dict: dict,
        mask_loss_scale: float,
        dice_loss_scale: float,
        cls_loss_scale: float,
        event_weight: torch.Tensor = None,
        loss_name: str = "segmentation"
):
    mask_loss, dice_loss, cls_loss = seg_loss_fn(
        predict_cls = predict_classification,
        predict_mask = predict_mask,
        target_cls = target_classification,
        target_mask = target_mask,
        class_weight = class_weight,
        segmentation_mask = segmentation_mask,
        point_cloud_mask = point_cloud_mask,
        reduction='none'
    )

    if event_weight is not None:
        mask_loss = mask_loss * event_weight
        dice_loss = dice_loss * event_weight
        cls_loss  = cls_loss * event_weight
        mask_loss = mask_loss.sum(dim=-1) / event_weight.sum(dim=-1).clamp(1e-6)
        dice_loss = dice_loss.sum(dim=-1) / event_weight.sum(dim=-1).clamp(1e-6)
        cls_loss = cls_loss.sum(dim=-1) / event_weight.sum(dim=-1).clamp(1e-6)
    else:# Sum over classes if multi-class
        cls_loss = cls_loss.mean()
        mask_loss = mask_loss.mean()
        dice_loss = dice_loss.mean()

    loss_dict[f"{loss_name}-cls"] = cls_loss
    loss_dict[f"{loss_name}-mask"] = mask_loss
    loss_dict[f"{loss_name}-dice"] = dice_loss

    loss = cls_loss * cls_loss_scale + mask_loss * mask_loss_scale + dice_loss * dice_loss_scale

    return loss


@time_decorator(name="[Segmentation] shared_epoch_end")
def shared_epoch_end(
):
    pass
