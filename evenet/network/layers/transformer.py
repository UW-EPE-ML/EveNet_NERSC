import torch.nn as nn
from torch import Tensor

from evenet.network.layers.utils import TalkingHeadAttention, StochasticDepth, LayerScale
from evenet.network.layers.linear_block import GRUGate, GRUBlock
from evenet.network.layers.activation import create_residual_connection


class TransformerBlockModule(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init,
                 drop_probability):
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
        # TransformerBlock input shapes: x: torch.Size([B, P, 128]), mask: torch.Size([B, P, 1])
        padding_mask = ~(mask.squeeze(2).bool()) if mask is not None else None  # [batch_size, num_objects]
        if self.talking_head:
            updates, _ = self.attn(self.norm1(x), int_matrix=None, mask=mask)
        else:
            updates, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                   key_padding_mask=padding_mask)

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
            x = x * mask

        return x


class GTrXL(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 hidden_dim_scale: float,
                 num_heads: int,
                 dropout: float):
        super(GTrXL, self).__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_gate = GRUGate(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = GRUBlock(input_dim=hidden_dim,
                                     hidden_dim_scale=hidden_dim_scale,
                                     output_dim=hidden_dim,
                                     normalization_type="LayerNorm",
                                     activation_type="gelu",
                                     dropout=dropout,
                                     skip_connection=True)

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        """

        :param x: (batch_size, num_objects, hidden_dim)
        :param padding_mask: (batch_size, num_objects)
        :param sequence_mask: (batch_size, num_objects, hidden_dim)
        :return:
        """
        output = self.attention_norm(x)
        output, _ = self.attention(
            output, output, output,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        output = self.attention_gate(output, x)

        return self.feed_forward(x=output, sequence_mask=sequence_mask)


class GatedTransformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 num_heads: int,
                 transformer_activation: str,
                 transformer_dim_scale: float,
                 dropout: float):
        super(GatedTransformer, self).__init__()
        self.num_layers = num_layers

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.transformer_activation = transformer_activation
        self.transformer_dim_scale = transformer_dim_scale

        self.layers = nn.ModuleList([
            GTrXL(hidden_dim=self.hidden_dim,
                  hidden_dim_scale=self.transformer_dim_scale,
                  num_heads=self.num_heads,
                  dropout=self.dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        """
        :param x: (batch_size, num_objects, hidden_dim)
        :param padding_mask: (batch_size, num_objects)
        :param sequence_mask: (batch_size, num_objects, hidden_dim)
        :return:
        """

        output = x

        for layer in self.layers:
            output = layer(
                x=output,
                padding_mask=padding_mask,
                sequence_mask=sequence_mask
            )

        return output


def create_transformer(
        transformer_type: str,
        num_layers: int,
        hidden_dim,
        num_heads,
        transformer_activation,
        transformer_dim_scale,
        dropout) -> nn.Module:
    """
    Create a transformer model with the specified options.

    :param options: Options for the transformer model.
    :param num_layers: Number of layers in the transformer.
    :return: Transformer model.
    """
    if transformer_type == "GatedTransformer":
        return GatedTransformer(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transformer_activation=transformer_activation,
            transformer_dim_scale=transformer_dim_scale,
            dropout=dropout
        )


class ClassifierTransformerBlockModule(nn.Module):
    def __init__(self,
                 input_dim: int,
                 projection_dim: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.bridge_class_token = create_residual_connection(
            skip_connection=True,
            input_dim=input_dim,
            output_dim=projection_dim
        )

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)
        self.norm3 = nn.LayerNorm(projection_dim)

        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Dropout(dropout),
            nn.Linear(2 * projection_dim, input_dim),
        )

    def forward(self, x, class_token, mask=None):
        """

        :param x: point_cloud (batch_size, num_objects, projection_dim)
        :param class_token: (batch_size, input_dim)
        :param mask: (batch_size, num_objects, 1)
        :return:
        """
        class_token = self.bridge_class_token(class_token)
        x1 = self.norm1(x)
        query = class_token.unsqueeze(1)  # Only use the class token as query
        padding_mask = ~(mask.unsqueeze(2).bool()) if mask is not None else None
        updates, _ = self.attn(query, x1, x1, key_padding_mask=padding_mask)  # [batch_size, 1, projection_dim]
        updates = self.norm2(updates)

        x2 = updates + query
        x3 = self.norm3(x2)
        cls_token = self.mlp(x3)

        return cls_token.squeeze(1)


class GeneratorTransformerBlockModule(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout, layer_scale, layer_scale_init, drop_probability):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_scale_flag = layer_scale
        self.drop_probability = drop_probability

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm3 = nn.LayerNorm(projection_dim)

        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        if layer_scale:
            self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
            self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

    def forward(self, x, cond_token, mask=None):
        x1 = self.norm1(x)
        updates, _ = self.attn(x1, x1, x1, key_padding_mask=~mask.bool() if mask is not None else None)

        if self.layer_scale_flag:
            updates = self.layer_scale1(updates, mask)
        x2 = updates + cond_token
        x3 = self.norm3(x2)
        x3 = self.mlp(x3)

        if self.layer_scale_flag:
            x3 = self.layer_scale2(x3, mask)
        cond_token = x2 + x3

        return x, cond_token
