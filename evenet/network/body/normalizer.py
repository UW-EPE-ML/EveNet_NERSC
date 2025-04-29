import torch
from torch import nn, Tensor
import math
from torch.distributions import Normal
from typing import List


class Normalizer(nn.Module):
    def __init__(self, mean: Tensor, std: Tensor, norm_mask: Tensor, inv_cdf_index=None):

        super(Normalizer, self).__init__()

        if inv_cdf_index is None:
            inv_cdf_index = []
        """
        :param
            log_mask: mask to apply before normalization. shape (num_features,)
            mean: mean value for normalization. shape (num_features,)
            std: standard deviation for normalization . shape (num_features,)
        """
        # Initialize mean and std as parameters
        self.register_buffer("norm_mask", norm_mask)
        mean = torch.where(self.norm_mask, mean, 0.0)
        std = torch.where(self.norm_mask, std, 1.0)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp(min=1e-6))
        self.inv_cdf_index = inv_cdf_index
        self.normal = Normal(0, 1)

        # self.log_mask_expanded = self.log_mask.unsqueeze(0).unsqueeze(0) if self.log_mask is not None else None

    @torch.no_grad()
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        :param x: input point cloud (batch_size, num_objects, num_features)
        :param mask: mask for point cloud (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
        :return: tensor (batch_size, num_objects, num_features)
        """
        # Apply the log mask to the input tensor
        # x = torch.where(self.log_mask_expanded, torch.log1p(x), x)  # log1p(x) = log(1 + x) to avoid log(0) issues # TODO
        x = (x - self.mean) / self.std
        if mask is not None:
            x = x * mask
        if len(self.inv_cdf_index) > 0:
            # After normalization, apply inverse CDF transformation
            x_partial = x[..., self.inv_cdf_index].contiguous()
            # normalized uniform: [-sqrt(3), sqrt(3)] , add extra 0.1 to avoid unperfect mean, std deviation
            x_partial = (x_partial + (math.sqrt(3) + 0.1)) / (2 * (math.sqrt(3) + 0.1))
            x_partial = torch.clamp(x_partial, 1e-6, 1 - 1e-6)
            x[..., self.inv_cdf_index] = self.normal.icdf(x_partial)
            if mask is not None:
                x = x * mask

        return x

    @torch.no_grad()
    def denormalize(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """

        :param x: input point cloud (batch_size, num_objects, num_features)
        :param mask: mask for point cloud (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
        :return: tensor (batch_size, num_objects, num_features)
        """
        if len(self.inv_cdf_index) > 0:
            x_partial = x[..., self.inv_cdf_index].contiguous()
            x_partial = self.normal.cdf(x_partial)
            x_partial = x_partial * 2 * (math.sqrt(3) + 0.1) - (math.sqrt(3) + 0.1)
            x[..., self.inv_cdf_index] = x_partial
            if mask is not None:
                x = x * mask
        x = (x * self.std) + self.mean
        # x = torch.where(self.log_mask_expanded, torch.expm1(x), x) # TODO
        if mask is not None:
            x = x * mask
        return x
