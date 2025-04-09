import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

from typing import Optional

from evenet.network.body.embedding import FourierEmbedding
from evenet.network.layers.activation import create_activation
from evenet.network.layers.linear_block import ResNetDense
from evenet.network.layers.utils import StochasticDepth
from evenet.network.layers.transformer import GeneratorTransformerBlockModule
class EventGenerationHead(nn.Module):
    def __init__(self,
                 input_dim: int,
                 projection_dim: int,
                 num_global_cond: int,
                 num_classes: int,
                 num_feat: int,
                 num_layers: int, simple,
                 num_heads: int,
                 dropout: float,
                 talking_head: bool,
                 layer_scale: bool,
                 layer_scale_init: float,
                 drop_probability: float,
                 feature_drop):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.num_diffusion = 3  # Adjust this based on your settings

        self.global_cond_embedding = nn.Sequential(
            nn.Linear(num_global_cond, projection_dim),
            nn.GELU(approximate='none')
        )
        self.time_embedding = FourierEmbedding(projection_dim)
        self.cond_token = nn.Sequential(
            nn.Linear(2 * projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2 * projection_dim, projection_dim),
            nn.GELU(approximate='none')
        )
        #self.label_embedding = nn.Embedding(num_classes, projection_dim)
        self.label_dense = nn.Linear(num_classes, projection_dim, bias=False)
        self.feature_drop = feature_drop
        self.stochastic_depth = StochasticDepth(feature_drop)
        self.gen_transformer_blocks = nn.ModuleList([
            GeneratorTransformerBlockModule(projection_dim, num_heads, dropout, talking_head,
                                            layer_scale, layer_scale_init, drop_probability)
            for _ in range(num_layers)
        ])
        self.generator = nn.Linear(projection_dim, num_feat)

    def forward(self, x, jet, mask, time, label):
        jet_emb = self.jet_embedding(jet)  # jet_emb shape after embedding: torch.Size([B, proj_dim])
        time_emb = self.time_embedding(time)  # time_emb shape after embedding: torch.Size([B, 1, proj_dim])
        time_emb = time_emb.squeeze(1)  # time_emb shape after squeezing: torch.Size([B, proj_dim])
        cond_token = self.cond_token(
            torch.cat([time_emb, jet_emb], dim=-1))  # After MLP, cond_token shape: torch.Size([B, proj_dim])

        if label is not None:
            # label_emb = self.label_embedding(label)
            label_emb = self.label_dense(label.float())
            label_emb = self.stochastic_depth(label_emb)
            cond_token = cond_token + label_emb
        else:
            print("ERROR: In Generation Head, Label is None, skipping label embedding")

        cond_token = cond_token.unsqueeze(1).expand(-1, x.shape[1], -1) * mask.unsqueeze(-1)

        for transformer_block in self.gen_transformer_blocks:
            concatenated = cond_token + x
            out_x, cond_token = transformer_block(concatenated, cond_token, mask)
        x = cond_token + x
        x = F.layer_norm(x, [x.size(-1)])
        x = self.generator(x)

        return x * mask.unsqueeze(-1)

class GlobalCondGenerationHead(nn.Module):

    def __init__(self,
                 num_layer: int,
                 num_resnet_layer: int,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 input_cond_dim: int,
                 num_classes: int,
                 resnet_dim: int,
                 layer_scale_init: float,
                 feature_drop_for_stochastic_depth: float,
                 activation: str,
                 dropout: float):
        super(GlobalCondGenerationHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp_dim = resnet_dim
        self.layer_scale_init = layer_scale_init
        self.dropout = dropout
        self.resnet_num_layer = num_resnet_layer
        self.activation = activation
        self.num_layer = num_layer
        self.num_classes = num_classes

        self.fourier_projection = FourierEmbedding(self.hidden_dim)
        self.dense_t = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )

        if input_cond_dim > 0:
            self.global_cond_embedding = nn.Sequential(
                nn.Linear(input_cond_dim, self.hidden_dim),
                create_activation(self.activation, self.hidden_dim)
            )
        if num_classes > 0:
            self.label_embedding = nn.Sequential(
                nn.Linear(num_classes, 2 * self.hidden_dim, bias=False),
                StochasticDepth(feature_drop_for_stochastic_depth)
            )

        self.cond_token_embedding = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            create_activation(self.activation, self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            create_activation(self.activation, self.hidden_dim)
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            create_activation(self.activation, self.hidden_dim)
        )

        self.resnet_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim if i == 0 else self.mlp_dim, eps=1e-6),
                ResNetDense(
                    input_dim = self.hidden_dim if i == 0 else self.mlp_dim,
                    hidden_dim = self.mlp_dim,
                    output_dim = self.mlp_dim,
                    num_layers = self.resnet_num_layer,
                    activation = self.activation,
                    dropout = self.dropout,
                    layer_scale_init = self.layer_scale_init
                )
            )
            for i in range(self.num_layer - 1)
        ])

        self.out_layer_norm = nn.LayerNorm(self.mlp_dim, eps=1e-6)
        self.out = nn.Linear(self.mlp_dim, output_dim)
        nn.init.zeros_(self.out.weight)

    def forward(self,
                x: Tensor,
                time: Tensor,
                x_mask: Optional[Tensor] = None,
                global_cond: Optional[Tensor] = None,
                label: Optional[Tensor] = None
        ) -> Tensor:
        # ----------------
        # x: [B, 1, D] <- Noised Global Input
        # x_mask: [B, 1, 1] <- Mask
        # t: [B, 1] <- Time
        # global_cond: [B, C] <- Global Condition
        # label: [B, 1] <- Conditional Label, one-hot in function
        # ----------------

        batch_size = x.shape[0]
        if x_mask is None:
            x_mask = torch.ones((batch_size, 1, 1), device=x.device)

        embed_time = self.fourier_projection(time).unsqueeze(1)  # [B, 1, D]
        # TODO: Add conditional labels
        if global_cond is not None:
            global_cond_token = self.global_cond_embedding(global_cond)  # [B, 1, D]
            global_token = torch.cat([global_cond_token, embed_time], dim=-1)  # [B, 1, 2D]
        else:
            global_token = self.dense_t(embed_time)  # [B, 1, 2D]

        cond_token = self.cond_token_embedding(global_token)  # [B, 1, 2D]

        if label is not None:
            label = F.one_hot(label, num_classes = self.num_classes).float().unsqueeze(1) # [B, 1, C]
            cond_label = self.label_embedding(label) # [B, 1, 2D]
            cond_token = cond_token + cond_label

        scale, shift = torch.chunk(cond_token, 2, dim=-1)  # [B, 1, D], [B, 1, D]

        embed_x = self.dense_layer(x)
        embed_x = (embed_x * (1.0 + scale) + shift) * x_mask

        for resnet_layer in self.resnet_layers:
            embed_x = resnet_layer(embed_x) * x_mask
        embed_x = self.out_layer_norm(embed_x) * x_mask
        outputs = self.out(embed_x) * x_mask

        return outputs.squeeze(1) # [B, output_dim]



