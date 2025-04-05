import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class ConfusionMatrixAccumulator:
    def __init__(self, num_classes, normalize=False):
        self.num_classes = num_classes
        self.normalize = normalize
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        self.matrix += cm

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reduce_across_gpus(self):
        """All-reduce across DDP workers"""
        if torch.distributed.is_initialized():
            tensor = torch.tensor(self.matrix, dtype=torch.long, device="cuda")
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            self.matrix = tensor.cpu().numpy()

    def compute(self):
        """Return normalized or raw matrix"""
        cm = self.matrix.astype(np.float64)
        if self.normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm / row_sums)
        return cm

    def plot(self, class_names=None, cmap="Blues", normalize=True):
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
