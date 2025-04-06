from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ClassificationMetrics:
    def __init__(self, num_classes, device, normalize=False, num_bins=50):
        self.device = device
        self.num_classes = num_classes
        self.normalize = normalize
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.valid = 0
        self.total = 0

        # for logits histogram
        self.bins = np.linspace(0, 1, num_bins + 1)
        self.hist_store = np.zeros((self.num_classes, self.num_classes, num_bins), dtype=np.int64)

    def update(self, y_true, y_pred_raw):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred_raw, torch.Tensor):
            y_pred = y_pred_raw.argmax(dim=-1).detach().cpu().numpy()
            logits = y_pred_raw.detach().cpu()

        # Filter out ignored targets like -1 (often used for masking)
        valid = y_true >= 0
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        self.valid += valid.sum()
        self.total += len(valid)

        if len(y_true) == 0:
            return  # Skip empty updates safely

        present_labels = np.unique(np.concatenate([y_true, y_pred]))
        cm_partial = confusion_matrix(y_true, y_pred, labels=present_labels)

        for i, true_label in enumerate(present_labels):
            for j, pred_label in enumerate(present_labels):
                if true_label < self.num_classes and pred_label < self.num_classes:
                    self.matrix[true_label, pred_label] += cm_partial[i, j]

        # For Logits histogram
        probs = F.softmax(logits, dim=1).numpy()
        for true_cls in range(self.num_classes):
            mask = (y_true == true_cls)
            if not np.any(mask):
                continue
            probs_true = probs[mask]  # (N_true, num_classes)

            for pred_cls in range(self.num_classes):
                scores = probs_true[:, pred_cls]
                hist, _ = np.histogram(scores, bins=self.bins)
                self.hist_store[true_cls, pred_cls] += hist

    def reset(self, cm: bool = True, logits: bool = True):
        self.valid = 0
        self.total = 0
        if cm:
            self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        if logits:
            self.hist_store = np.zeros((self.num_classes, self.num_classes, self.bins.size - 1), dtype=np.int64)

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

    def compute(self):
        """Return normalized or raw matrix"""
        cm = self.matrix.astype(np.float64)
        if self.normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm / row_sums)
        return cm

    def plot_cm(self, class_names=None, cmap="Blues", normalize=True):
        cm = self.compute() if normalize else self.matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(self.num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names or tick_marks, rotation=45)
        ax.set_yticklabels(class_names or tick_marks)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        return fig

    def plot_logits(self, class_names, train_hist_store=None):
        results = {}

        # Use training store if provided, else default to self.hist_store
        if train_hist_store is None:
            train_hist_store = self.hist_store

        # Custom color palette (max 10 classes)
        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]

        bin_centers = 0.5 * (self.bins[1:] + self.bins[:-1])
        bin_widths = np.diff(self.bins)

        for true_cls in range(self.num_classes):
            fig = plt.figure(figsize=(10, 8))

            for cls in range(self.num_classes):
                # Plot training histogram (bars)
                train_counts = train_hist_store[true_cls, cls]
                if np.sum(train_counts) > 0:
                    train_density = train_counts / (np.sum(train_counts) * bin_widths)
                    color = colors[cls % len(colors)]
                    label = f"{class_names[cls]} (True, Train)" if cls == true_cls else f"{class_names[cls]} (Train)"

                    if cls == true_cls:
                        plt.bar(bin_centers, train_density, width=bin_widths, color=color, alpha=0.85, label=None,
                                edgecolor='black')
                    else:
                        plt.bar(bin_centers, train_density, width=bin_widths, color=color, alpha=0.7,
                                label=None, edgecolor=color, fill=False)

                # Plot validation histogram (lines with markers)
                val_counts = self.hist_store[true_cls, cls]
                if np.sum(val_counts) > 0:
                    val_density = val_counts / (np.sum(val_counts) * bin_widths)
                    color = colors[cls % len(colors)]
                    label = f"{class_names[cls]} (True)" if cls == true_cls else f"{class_names[cls]}"
                    plt.plot(
                        bin_centers, val_density, color=color, label=label,
                        linestyle='-' if cls == true_cls else '--',
                        marker='o' if cls == true_cls else 'x',
                        linewitdh=2 if cls == true_cls else 1,
                    )

            title = f"True Class {class_names[true_cls] if class_names else true_cls}"
            plt.title(f"{title}: Softmax Score Distribution (Train vs Val)")
            plt.xlabel("Softmax Score")
            plt.ylabel("Density")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()

            results[true_cls] = fig

        return results
