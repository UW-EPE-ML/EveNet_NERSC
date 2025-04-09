import torch
from torch import Tensor

def loss(predict: Tensor,
         target: Tensor):

    return torch.mean((predict - target) ** 2)