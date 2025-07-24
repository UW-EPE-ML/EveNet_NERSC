import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

from typing import Optional

from evenet.network.layers.transformer import SegmentationTransformerBlockModule

class MHAttentionMapPointCloud(nn.Module):
    """Multi-head self-attention map for point clouds. Returns only attention weights."""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)

        self.normalize_fact = float(hidden_dim / num_heads) ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        q: Tensor [B, Q, F]   # Q query vectors per batch (could be same as N)
        k: Tensor [B, N, F]   # N key vectors (e.g. same as input point cloud)
        mask: Tensor [B, N] (optional) - masks out attention for certain points
        """
        B, Q, _ = q.shape
        N = k.shape[1]

        q = self.q_linear(q)  # [B, Q, H]
        k = self.k_linear(k)  # [B, N, H]

        qh = q.view(B, Q, self.num_heads, self.hidden_dim // self.num_heads)  # [B, Q, Hn, Dh]
        kh = k.view(B, N, self.num_heads, self.hidden_dim // self.num_heads)  # [B, N, Hn, Dh]

        # Attention scores: [B, Q, Hn, N]
        attn = torch.einsum("bqhd,bnhd->bqnh", qh * self.normalize_fact, kh) # [B, Q, N, Hn]

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(-1), float("-inf"))

        attn = F.softmax(attn.flatten(2), dim=-1).view(attn.size())  # over N
        attn = self.dropout(attn)  # [B, Q, Hn, N]

        return attn

class MaskHead(nn.Module):
    def __init__(
            self,
            projection_dim: int,
            num_heads: int,
            dropout: float,
            dim_decay_rate: float = 2.0
        ):
        """
        "MaskHead for segmentation tasks.
        """
        super(MaskHead, self).__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_decay_rate = dim_decay_rate

        self.mlp = self._build_mlp()

    def _build_mlp(self):
        layers = []
        in_dim = self.projection_dim + self.num_heads  # projection_dim + num_heads for attention map

        # First pxp projection (identity projection)
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(self.dropout))

        while in_dim > 1:
            out_dim = max(1, int(in_dim / self.dim_decay_rate))
            layers.append(nn.LayerNorm(in_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            if in_dim == 1:
                break

        return nn.Sequential(*layers)



    def forward(self, q: Tensor, bbox_mask: Tensor):
        """
        Args:
            q: (batch_size, num_patches, projection_dim)
            bbox_mask: (batch_size, num_queries, num_patches, num_heads) - mask for attention
        """
        q_expand = q.unsqueeze(1).repeat(1, bbox_mask.shape[1], 1, 1)  # (batch_size, num_queries, num_patches, projection_dim)
        x = torch.cat([q_expand, bbox_mask], dim=-1)  # (batch_size, num_queries, num_patches,  projection_dim + num_heads)
        x = x.flatten(0, 1)  # (batch_size * num_queries, num_patches, projection_dim + num_head)
        x = self.mlp(x)  # (batch_size * num_queries, num_patches, 1)
        x = x.squeeze(-1)  # (batch_size * num_queries, num_patches)

        x = x.view(-1, bbox_mask.shape[1], bbox_mask.shape[2])  # (batch_size, num_queries, num_patches)

        return x



class SegmentationTransformerDecoder(nn.Module):
    def __init__(
            self,
            projection_dim: int,
            num_heads: int,
            dropout: float,
            num_layers: int,
            return_intermediate: bool = False,
    ):
        super(SegmentationTransformerDecoder, self).__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.layers = nn.ModuleList([
            SegmentationTransformerBlockModule(
                projection_dim = self.projection_dim,
                num_heads = self.num_heads,
                dropout = self.dropout,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(self.projection_dim)



    def forward(self,tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None
                ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output, memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos
            )
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate) # (num_layers, batch_size, num_queries, projection_dim)
        return output.unsqueeze(0)  # (1, batch_size, num_queries, projection_dim)


class SegmentationHead(nn.Module):
    def __init__(self,
        projection_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        num_class: int = 1,  # Binary classification for mask prediction
        num_queries: int = 5,
        return_intermediate: bool = False,
        dim_decay_rate: float = 4.0
    ):
        super(SegmentationHead, self).__init__()

        self.return_intermediate = return_intermediate
        self.decoder = SegmentationTransformerDecoder(
            projection_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            return_intermediate=return_intermediate,
        )
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, projection_dim)

        self.mask_attention = MHAttentionMapPointCloud(
            query_dim=projection_dim,
            hidden_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.mask_decoder = MaskHead(
            projection_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            dim_decay_rate=dim_decay_rate
        )

        self.class_embed = nn.Linear(projection_dim, num_class + 1)  # Binary classification for mask prediction


    def forward(self, memory, memory_mask,
                pos_embed: Optional[Tensor] = None,
                **kwargs
        ):
        """
        Args:
            memory: (batch_size, num_patches, projection_dim)
            memory_mask: (batch_size, num_patches, 1)
            **kwargs: additional arguments for the decoder
        """
        batch_size, num_patches, projection_dim = memory.shape
        padding_mask = ~(memory_mask.squeeze(-1).bool())  if memory_mask is not None else None # (batch_size, num_patches)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, num_queries, projection_dim)
        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(
            tgt, memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=padding_mask,
            pos=pos_embed,
            query_pos=query_embed
        ) # (num_layers, batch_size, num_queries, projection_dim)

        mask_attention = self.mask_attention(
            q = hs[-1],
            k = memory,
            mask = padding_mask
        )

        pred_mask = self.mask_decoder(
            q = memory,
            bbox_mask = mask_attention
        ) # (B, N, P)

        memory_mask.squeeze(-1).unsqueeze(1).repeat(1, self.num_queries, 1)
        pred_mask = pred_mask * memory_mask

        pred_class = self.class_embed(hs[-1])

        return pred_class, pred_mask

