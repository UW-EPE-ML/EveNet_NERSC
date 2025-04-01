import torch
from torch import Tensor

from evenet.dataset.regressions.base_regression import Regression, Statistics
from evenet.dataset.types import Source

class GaussianRegression(Regression):
    @staticmethod
    def name():
        return "gaussian"

    @staticmethod
    def statistics(source: Source) -> Statistics:
        data = source.data[source.mask]
        mean = torch.nanmean(data)
        std = torch.sqrt(torch.nanmean(torch.square(data)) - torch.square(mean))

        return Statistics(mean, std)

    @staticmethod
    def loss(predictions: Tensor, targets: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return torch.square((predictions - targets) / std)
