from abc import ABC, abstractmethod
from torch import Tensor

from evenet.dataset.types import Statistics
from evenet.dataset.types import Source

class Regression(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def statistics(source: Source) -> Statistics:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def loss(predictions: Tensor, targets: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        raise NotImplementedError()

    @staticmethod
    def normalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return (data - mean) / std

    @staticmethod
    def denormalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return std * data + mean
