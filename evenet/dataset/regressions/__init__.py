from typing import List, Dict, Type, Callable
from torch import Tensor

from evenet.dataset.regressions.base_regression import Regression
from evenet.dataset.types import Statistics
from evenet.dataset.regressions.gaussian_regression import GaussianRegression
from evenet.dataset.regressions.laplacian_regression import LaplacianRegression
from evenet.dataset.regressions.log_gaussian_regression import LogGaussianRegression


all_regressions: List[Type[Regression]] = [
    GaussianRegression,
    LaplacianRegression,
    LogGaussianRegression
]

regression_mapping: Dict[str, Type[Regression]] = {
    regression.name(): regression
    for regression in all_regressions
}


def regression_class(name: str) -> Type[Regression]:
    return regression_mapping[name]


def regression_loss(name: str) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    return regression_mapping[name].loss


def regression_statistics(name: str) -> Callable[[Tensor], Statistics]:
    return regression_mapping[name].statistics

