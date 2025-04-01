import torch
from torch import Tensor

from evenet.dataset.regressions.base_regression import Regression, Statistics
from evenet.dataset.types import Source

class LaplacianRegression(Regression):
    @staticmethod
    def name():
        return "laplacian"

    @staticmethod
    def statistics(source: Source) -> Statistics:
        data = source.data[source.mask]
        valid_data = data[~torch.isnan(data)]

        median = torch.median(valid_data)
        deviation = torch.mean(torch.abs(valid_data - median))

        return Statistics(median, deviation)

    @staticmethod
    def loss(predictions: Tensor, targets: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return torch.abs(predictions - targets) / std
