import torch
import torch.nn as nn

from evenet.network_scratch.layers.utils import TalkingHeadAttention, StochasticDepth, LayerScale
class TransformerBlockModule(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init, drop_probability):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.talking_head = talking_head
        self.layer_scale_flag = layer_scale
        self.drop_probability = drop_probability

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)

        if talking_head:
            self.attn = TalkingHeadAttention(projection_dim, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Dropout(dropout),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        self.drop_path = StochasticDepth(drop_probability)

        if layer_scale:
            self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
            self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

    def forward(self, x, mask):
        # TransformerBlock input shapes: x: torch.Size([B, P, 128]), mask: torch.Size([B, P])
        if self.talking_head:
            updates, _ = self.attn(self.norm1(x), int_matrix=None, mask=mask)
        else:
            updates, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=~mask.bool() if mask is not None else None)

        if self.layer_scale_flag:
            # Input updates: torch.Size([B, P, 128]), mask: torch.Size([B, P])
            x2 = x + self.drop_path(self.layer_scale1(updates, mask))
            x3 = self.norm2(x2)
            x = x2 + self.drop_path(self.layer_scale2(self.mlp(x3), mask))
        else:
            x2 = x + self.drop_path(updates)
            x3 = self.norm2(x2)
            x = x2 + self.drop_path(self.mlp(x3))

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x

