import torch
from torch import nn, Tensor
from evenet.network.layers.linear_block.regularization import StochasticDepth, LayerScale

from evenet.network.layers.embedding.time_embedding import FourierProjection
from evenet.control.config import DotDict


class ResNetDense(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=2, dropout=0.0, layer_scale_init=1.0):
        super(ResNetDense, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.layer_scale_init = layer_scale_init

        # Define the layers
        self.residual_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.SiLU(), nn.Dropout(self.dropout)) for i in range(nlayers)
        ])
        self.layer_scale = LayerScale(self.layer_scale_init, hidden_size)

    def forward(self, x):
        residual = self.residual_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.layer_scale(x)
        return residual + x


class TimeConditionedResNet(nn.Module):

    def __init__(self, options: DotDict, num_layer: int, input_dim: int, projection_dim: int, mlp_dim: int):
        super(TimeConditionedResNet, self).__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.mlp_dim = mlp_dim
        self.layer_scale_init = options.Network.PET_layer_scale_init
        self.dropout = options.Network.PET_dropout
        self.sub_resnet_nlayer = options.Network.diff_sub_resnet_nlayer

        self.fourier_projection = FourierProjection(self.projection_dim)
        self.dense_t = nn.Sequential(nn.Linear(self.projection_dim, 2 * self.projection_dim), nn.GELU())
        self.dense_layer = nn.Sequential(nn.Linear(self.input_dim, self.projection_dim), nn.SiLU())
        self.resnet_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.projection_dim if i == 0 else mlp_dim, eps=1e-6),
                ResNetDense(self.projection_dim if i == 0 else mlp_dim, mlp_dim, num_layer, self.dropout,
                            self.layer_scale_init))
            for i in range(num_layer - 1)
        ]
        )
        self.out_layer_norm = nn.LayerNorm(mlp_dim, eps=1e-6)
        self.out = nn.Linear(mlp_dim, input_dim)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        # ----------------
        # x: [B, 1, D] <- Global Input
        # t: [B, 1]
        # label: [B, C]
        # ----------------

        embed_time = self.fourier_projection(time)
        # TODO: Add conditional labels
        embed_time = self.dense_t(embed_time).unsqueeze(1)  # [B, 1, 2D]
        scale, shift = torch.chunk(embed_time, 2, dim=-1)  # [B, 1, D], [B, 1, D]

        embed_x = self.dense_layer(x)
        embed_x = embed_x * (1.0 + scale) + shift

        for resnet_layer in self.resnet_layers:
            embed_x = resnet_layer(embed_x)
        embed_x = self.out_layer_norm(embed_x)
        outputs = self.out(embed_x)

        return outputs
