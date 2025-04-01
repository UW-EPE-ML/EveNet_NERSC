import torch
from torch import nn, Tensor

from evenet.network.layers.linear_block.masking import create_masking

# Implementing StochasticDepth, LayerScale, and TalkingHeadAttention as described earlier
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        # --------------
        # x: [B, T, D]
        # --------------
        if self.training:
            keep_prob = 1 - self.drop_prob
            # Creating a tensor of shape [batch_size, 1, 1, ...] to match x's shape
            random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
            # Random tensor values greater than 1 are set to 1, otherwise 0
            random_tensor = torch.floor(random_tensor)
            return x * random_tensor
        return x


class LayerScale(nn.Module):
    def __init__(self, init_values, projection_dim):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(projection_dim))

    def forward(self, x, mask=None):
        if mask is not None:
            return x * self.gamma * mask.unsqueeze(-1)
        else:
            return x * self.gamma


class RandomDrop(nn.Module):
    def __init__(self, drop_prob: float, num_skip: int):
        super().__init__()
        self.drop_prob = drop_prob
        self.num_skip = num_skip

    def forward(self, x):
        if not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize [ IN-PLACE OPERATION ]

        # Create a new tensor instead of modifying in-place
        output = x.clone()
        output[:, :, self.num_skip:] = x[:, :, self.num_skip:] * random_tensor.unsqueeze(2)
        return output


