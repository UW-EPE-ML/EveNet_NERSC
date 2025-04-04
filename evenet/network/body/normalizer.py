import torch
from torch import nn, Tensor


class Normalizer(nn.Module):
    def __init__(self, log_mask: Tensor, mean: Tensor, std: Tensor):
        super(Normalizer, self).__init__()

        """
        :param
            log_mask: mask to apply before normalization. shape (num_features,)
            mean: mean value for normalization. shape (num_features,)
            std: standard deviation for normalization . shape (num_features,)
        """
        # Initialize mean and std as parameters
        self.log_mask = nn.Parameter(log_mask, requires_grad=False)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.log_mask_expanded = self.log_mask.unsqueeze(0).unsqueeze(0)

        print("log_mask_expanded.device:", self.log_mask_expanded.device, "log_mask.device:", self.log_mask.device)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        :param x: input point cloud (batch_size, num_objects, num_features)
        :param mask: mask for point cloud (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
        :return: tensor (batch_size, num_objects, num_features)
        """
        # Apply the log mask to the input tensor
        x = torch.where(self.log_mask_expanded, torch.log1p(x), x)  # log1p(x) = log(1 + x) to avoid log(0) issues
        x = (x - self.mean) / self.std
        if mask is not None:
            x = x * mask
        return x

    def denormalize(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """

        :param x: input point cloud (batch_size, num_objects, num_features)
        :param mask: mask for point cloud (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
        :return: tensor (batch_size, num_objects, num_features)
        """
        print("x.device:", x.device)
        print("log_mask_expanded.device:", self.log_mask_expanded.device)

        x = (x * self.std) + self.mean
        x = torch.where(self.log_mask_expanded, torch.expm1(x), x)
        if mask is not None:
            x = x * mask
        return x
