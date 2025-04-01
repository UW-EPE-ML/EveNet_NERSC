import torch
from torch import nn, Tensor
import torch.nn.functional as F

from evenet.control.config import DotDict, EventInfo
from evenet.network.layers.linear_block.masking import create_masking
from evenet.network.layers.linear_block.regularization import StochasticDepth, LayerScale
from evenet.network.layers.embedding.time_embedding import FourierProjection


class GeneratorTransformerBlock(nn.Module):
    def __init__(self, options: DotDict, extra_dim: int = 0):
        super(GeneratorTransformerBlock, self).__init__()
        self.projection_dim = options.Network.hidden_dim
        self.num_heads = options.Network.PET_num_heads
        self.drop_probability = options.Network.PET_drop_probability
        self.dropout = options.Network.PET_dropout
        self.layer_scale = options.Network.PET_layer_scale
        self.layer_scale_init = options.Network.PET_layer_scale_init
        self.talking_head = options.Network.PET_talking_head

        self.group_norm1 = nn.LayerNorm(self.projection_dim)
        self.group_norm2 = nn.LayerNorm(self.projection_dim)
        self.dense1 = nn.Sequential(nn.Linear(self.projection_dim, 2 * self.projection_dim), nn.GELU())
        self.dense2 = nn.Linear(2 * self.projection_dim, self.projection_dim)
        self.dropout_block = nn.Dropout(self.dropout)
        self.multihead_attn = nn.MultiheadAttention(self.projection_dim, self.num_heads)
        self.layer_scale_fn1 = LayerScale(self.layer_scale_init, self.projection_dim)
        self.layer_scale_fn2 = LayerScale(self.layer_scale_init, self.projection_dim)
        self.masking = create_masking(options.Network.masking)

    def forward(self, encoded: Tensor, cond_token: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:

        # -------------------------
        # Transformer block for PET module
        # encoded: [T, B, D]
        # cond_token: [T, B, D]
        # padding_mask: [T, B] (False means padding)
        # sequence_mask: [T, B, D] (False means padding)
        # -------------------------
        # output: [T, B, D]
        # -------------------------

        concatenated = cond_token + encoded

        x1 = self.group_norm1(concatenated)  # [T, B, D]
        updates, _ = self.multihead_attn(x1, x1, x1, key_padding_mask=padding_mask)

        if self.layer_scale:
            updates = self.masking(self.layer_scale_fn1(updates), sequence_mask)  # [T, B, D]

        x2 = updates + cond_token

        x3 = self.group_norm2(x2)
        x3 = self.dense1(x3)
        x3 = self.dense2(x3)  # [T, B, D]

        if self.layer_scale:
            x3 = self.masking(self.layer_scale_fn2(x3), sequence_mask)

        cond_token = x3 + x2

        return cond_token


class FeatureGenerationDecoder(nn.Module):
    def __init__(self, options: DotDict, cond_dim: int, out_dim: int, extra_dim: int = 0):
        super(FeatureGenerationDecoder, self).__init__()
        self.masking = create_masking(options.Network.masking)
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.projection_dim = options.Network.hidden_dim
        self.drop_probability = options.Network.PET_drop_probability
        self.num_layers = options.Network.diff_transformer_nlayer
        self.fourier_projection = FourierProjection(self.projection_dim)
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_dim, self.projection_dim),
            nn.GELU()
        )
        self.cond_dense1 = nn.Sequential(
            nn.Linear(2 * self.projection_dim, 2 * self.projection_dim),
            nn.GELU(),
            nn.Linear(2 * self.projection_dim, self.projection_dim),
            nn.GELU()
        )

        #    self.label_dense    = nn.Linear(num_classes, self.projection_dim, bias = False)

        self.stochastic_depth = StochasticDepth(self.drop_probability)
        self.transformers = nn.ModuleList(
            [GeneratorTransformerBlock(options, extra_dim) for i in range(self.num_layers)])
        self.group_norm1 = nn.LayerNorm(self.projection_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dense_out = nn.Linear(self.projection_dim, self.out_dim)

    def forward(self, x: Tensor, cond: Tensor, time: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        # -------------------------
        # Transformer block for PET module
        # x: [T, B, D]
        # cond: [B, D1]
        # lable: [B, C]
        # time: [B, 1]
        # padding_mask: [T, B] (False means padding)
        # sequence_mask: [T, B, 1] (False means padding)
        # -------------------------
        # output: [T, B, D]
        # -------------------------

        embed_time = self.fourier_projection(time)
        cond_jet = self.cond_embedding(cond)
        cond_token = torch.cat((embed_time, cond_jet), dim=-1)
        cond_token = self.cond_dense1(cond_token)

        #    cond_label = self.label_dense(label)
        #    cond_label = self.stochastic_depth(cond_label)
        #    cond_token = cond_token + cond_label
        cond_token = self.masking(cond_token.unsqueeze(0).repeat(x.shape[0], 1, 1), sequence_mask)

        for transformer in self.transformers:
            cond_token = transformer(x, cond_token, padding_mask, sequence_mask)

        encoded = self.group_norm1(x + cond_token)
        encoded = self.global_avg_pool(encoded.permute(1, 2, 0)).squeeze(-1)  # [T, B, D] -> [B, D, T] -> [B, D]
        encoded = self.dense_out(encoded)  # [B, O]

        return encoded
