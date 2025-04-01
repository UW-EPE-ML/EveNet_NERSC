from torch import nn, Tensor

class Normalizer(nn.Module):
    def __init__(self, mean: Tensor, std: Tensor):
        super(Normalizer, self).__init__()

        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, data: Tensor, mask: Tensor = None) -> Tensor:
        data = (data - self.mean) / self.std
        if mask is not None:
            return data * mask.unsqueeze(-1)
        else:
            return data

    def denormalize(self, data:Tensor, mask: Tensor = None) -> Tensor:
        data = data * self.std + self.mean
        if mask is not None:
            return data * mask.unsqueeze(-1)
        else: 
            return data
