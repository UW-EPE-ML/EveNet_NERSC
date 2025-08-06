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
from evenet.network.loss.segmentation import hungarian_matching
from sklearn.metrics import confusion_matrix, roc_curve, auc


class SegmentationMetrics:
    def __init__(
        self,
        device,
        hist_xmin: float = -0.5,
        hist_xmax: float = 5.5,
        num_bins: int = 6,
        mask_threshold: float = 0.5,
        process: int = "",
        clusters_label = None
    ):
        self.device = device
        self.hist_xmin = hist_xmin
        self.hist_xmax = hist_xmax
        self.num_bins = num_bins
        self.mask_threshold = mask_threshold
        self.process = process
        self.clusters_label = clusters_label
        self.num_classes = len(clusters_label)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.train_matrix = None
        self.labels = np.arange(self.num_classes)

    def update(
            self,
            y_true_mask: torch.Tensor,
            y_true_cls: torch.Tensor,
            y_pred_mask: torch.Tensor,
            y_pred_cls: torch.Tensor,
            segmentation_mask: torch.Tensor = None,
    ):
        # y_pred_mask = (y_pred_mask.sigmoid() > self.mask_threshold).float() # (B, N, P)
        # max_idx = y_pred_mask.argmax(dim=1, keepdim=True)  # shape: (B, 1, P)
        # # Create a mask with 1 at max index, 0 elsewhere
        # mask = torch.zeros_like(y_pred_mask)
        # mask.scatter_(1, max_idx, 1.0)
        # y_pred_mask = y_pred_mask * mask  # Apply the mask to y_pred_mask

        pred_indices, tgt_indices = hungarian_matching(
            predict_cls = y_pred_cls,
            predict_mask = y_pred_mask,
            target_cls = y_true_cls,
            target_mask = y_true_mask.float(),
            # segmentation_mask = segmentation_mask,
            include_cls_cost = False,
        )

        B, N_match = pred_indices.shape
        batch_idx = torch.arange(B, device=pred_indices.device).unsqueeze(-1)  # (B, 1)

        y_true_cls = y_true_cls[batch_idx, tgt_indices]
        y_pred_cls = y_pred_cls[batch_idx, pred_indices]
        y_true_mask = y_true_mask[batch_idx, tgt_indices]
        y_pred_mask = y_pred_mask[batch_idx, pred_indices]
        segmentation_mask = segmentation_mask[batch_idx, tgt_indices] if segmentation_mask is not None else None

        y_true_cls = y_true_cls.argmax(dim=-1).flatten().cpu().numpy()
        y_pred_cls = y_pred_cls.argmax(dim=-1).flatten().cpu().numpy()


        cm_partial = confusion_matrix(y_true_cls, y_pred_cls, labels=self.labels)
        for i, true_label in enumerate(self.labels):
            for j, pred_label in enumerate(self.labels):
                if true_label < self.num_classes and pred_label < self.num_classes:
                    self.matrix[true_label, pred_label] += cm_partial[i, j]

    def reset(self, cm: bool = True):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reduce_across_gpus(self):
        """All-reduce across DDP workers"""
        if torch.distributed.is_initialized():
            tensor = torch.tensor(self.matrix, dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            self.matrix = tensor.cpu().numpy()

    def assign_train_result(self, train_matrix=None):
        self.train_matrix = train_matrix

    def compute(self, matrix=None, normalize=False):
        """Return normalized or raw matrix"""
        cm = matrix.astype(np.float64)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm / row_sums)
        return cm

    def plot_cm(self, normalize=True):
        # --- Teal-Navy gradient colormap ---
        class_names = self.clusters_label.keys()
        gradient_colors = ('#f0f9fa', "#4ca1af")
        cmap = mcolors.LinearSegmentedColormap.from_list("teal_navy", gradient_colors)

        # --- Text colors for contrast ---
        text_colors = {
            "train_light": "#1E6B74",
            "train_dark": "#70E1E1",
            "valid_light": "#832424",
            "valid_dark": "#FFB4A2"
        }

        cm_valid = self.compute(self.matrix, normalize=normalize) if normalize else self.matrix

        # Optional: Compute train confusion matrix
        cm_train = None
        if self.train_matrix is not None:
            cm_train = self.compute(self.matrix, normalize=normalize) if normalize else self.train_matrix

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_valid, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(self.num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names or tick_marks, rotation=45, ha="right")
        ax.set_yticklabels(class_names or tick_marks)

        fmt = ".2f" if normalize else "d"

        for i in range(self.num_classes):
            for j in range(self.num_classes):

                cell_val = cm_valid[i, j]
                bg_val = cell_val / cm_valid.max()  # normalized background for contrast logic

                # Choose adaptive colors
                train_color = text_colors["train_dark"] if bg_val > 0.5 else text_colors["train_light"]
                valid_color = text_colors["valid_dark"] if bg_val > 0.5 else text_colors["valid_light"]

                y_offset = 0.15 if cm_train is not None else 0.0

                if cm_train is not None:
                    ax.text(j, i - y_offset, format(cm_train[i, j], fmt),
                            ha="center", va="center", color=train_color, fontsize=11)
                    ax.text(j, i + y_offset, format(cm_valid[i, j], fmt),
                            ha="center", va="center", color=valid_color, fontsize=11)
                else:
                    ax.text(j, i, format(cm_valid[i, j], fmt),
                            ha="center", va="center", color=valid_color, fontsize=11)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix (Train in Red, Valid in Black)")
        fig.tight_layout()
        return fig




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
        metrics: SegmentationMetrics,
        loss_dict: dict,
        mask_loss_scale: float,
        dice_loss_scale: float,
        cls_loss_scale: float,
        event_weight: torch.Tensor = None,
        loss_name: str = "segmentation",
        update_metrics: bool = True,
):

    # Null class don't need to be predicted, so we mask it out
    # predict_class_label = predict_classification.argmax(dim=-1)
    # class_zero_mask = (predict_class_label == 0)  # (B, N)
    # mask_expanded = class_zero_mask.unsqueeze(-1).expand(-1, -1, predict_mask.shape[-1]) # (B, N, P)
    # predict_mask = predict_mask.masked_fill(mask_expanded, -99999)  # Apply class zero mask to the predicted mask

    # print("class_zero_mask", mask_expanded.shape, "predict_class_label", predict_mask.shape)

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

    # print("predict_mask", predict_mask, "predict_cls", predict_classification)

    if update_metrics:
        metrics.update(
            y_true_mask=target_mask,
            y_true_cls=target_classification,
            y_pred_mask=predict_mask,
            y_pred_cls=predict_classification,
            segmentation_mask=segmentation_mask
        )

    loss_dict[f"{loss_name}-cls"] = cls_loss
    loss_dict[f"{loss_name}-mask"] = mask_loss
    loss_dict[f"{loss_name}-dice"] = dice_loss

    loss = cls_loss * cls_loss_scale + mask_loss * mask_loss_scale + dice_loss * dice_loss_scale

    return loss


@time_decorator(name="[Segmentation] shared_epoch_end")
def shared_epoch_end(
    global_rank,
    metrics_valid: SegmentationMetrics,
    metrics_train: SegmentationMetrics,
    logger,
    prefix: str = "",
):
    metrics_valid.reduce_across_gpus()
    if metrics_train is not None:
        metrics_train.reduce_across_gpus()
    if global_rank == 0:
        metrics_valid.assign_train_result(metrics_train.matrix if metrics_train is not None else None)
        fig_cm = metrics_valid.plot_cm(normalize=True)
        logger.log({
            f"{prefix}segmentation/CM": wandb.Image(fig_cm),
        })

    metrics_valid.reset()
    if metrics_train is not None:
        metrics_train.reset()


