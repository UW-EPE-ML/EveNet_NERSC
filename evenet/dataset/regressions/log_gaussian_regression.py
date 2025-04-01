import torch
from torch import Tensor

from evenet.dataset.regressions.base_regression import Regression, Statistics
from evenet.dataset.types import Source

class LogGaussianRegression(Regression):
    @staticmethod
    def name():
        return "log_gaussian"

    @staticmethod
    def signed_log(x: Tensor) -> Tensor:
        return torch.arcsinh(x / 2.0)

    @staticmethod
    def inverse_signed_log(x: Tensor) -> Tensor:
        return 2.0 * torch.sinh(x)

    @staticmethod
    def statistics(source: Source) -> Statistics:
        data = source.data[source.mask]
        data = LogGaussianRegression.signed_log(data)

        mean = torch.nanmean(data)
        std = torch.sqrt(torch.nanmean(torch.square(data)) - torch.square(mean))

        return Statistics(mean, std)

    @staticmethod
    def loss(predictions: Tensor, targets: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return torch.square(predictions - targets)

    @staticmethod
    def normalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        data = LogGaussianRegression.signed_log(data)
        return (data - mean) / std

    @staticmethod
    def denormalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        data = std * data + mean
        return LogGaussianRegression.inverse_signed_log(data)
