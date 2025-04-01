from typing import Tuple

import torch
from torch import Tensor, nn

from evenet.control.config import DotDict
from evenet.network.layers.linear_stack import create_linear_stack
from evenet.network.layers.transformer.PET_transformer import PointEdgeTransformer
from evenet.network.layers.embedding.time_embedding import FourierProjection
from evenet.network.layers.linear_block.masking import create_masking


class JetGenerationDecoder(nn.Module):
    def __init__(self, options: DotDict, input_dim: int, global_dim: int):
        super(JetGenerationDecoder, self).__init__()

        self.projection_dim = options.Network.hidden_dim

        self.encoder = PointEdgeTransformer(options, options.Network.diff_transformer_nlayer,
                                            2)  # TODO: Understand extra dim
        self.fourier_projection = FourierProjection(self.projection_dim)
        self.dense_t = nn.Linear(self.projection_dim, self.projection_dim)

        self.masking = create_masking(options.Network.masking)
        self.group_norm = nn.GroupNorm(1, options.nMaxJet)
        self.out = nn.Linear(self.projection_dim, input_dim)

    def forward(self, encoded_vectors: Tensor, source_time: Tensor, padding_mask: Tensor, sequence_mask: Tensor,
                global_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # -----------------------
        #  Parameters
        #  -----------------
        #  encoded_vectors: [T, B, D]
        #     Input sequence to predict on
        #  padding_mask: [B, T]
        #     Negative mask for transformer input
        #  sequence_mask: [T, B, 1]
        #     Positive mask for zeroing out padded vectors between operations.
        #  global_mask:   [T]
        #     Negative mask for indicating a sequential variable or a global variable.
        # -------------------------
        #  Output
        #  --------------------
        #  prediction_score: [B, ST, H]
        #     Prediction of score for sequential vectors
        #  sequential_padding_mask: [B, ST, 1]
        #     Positive mask for zeroing out padded vectors between operations.
        #  output_tensor: [B, T, H]
        # -----------------------------

        num_vectors, batch_size, hidden_dim = encoded_vectors.shape

        embed_time = self.fourier_projection(source_time).unsqueeze(0)  # [1, B, D]
        embed_time = self.dense_t(embed_time)
        combined_vectors = torch.cat((embed_time, encoded_vectors), dim=0)

        embed_time_padding_mask = padding_mask.new_zeros(batch_size, 1)  # [B, 1]
        combined_padding_mask = torch.cat((embed_time_padding_mask, padding_mask), dim=1)  # [B, T+1]

        embed_time_sequence_mask = sequence_mask.new_ones(1, batch_size, 1, dtype=torch.bool)  # [1, B, 1]
        combined_sequence_mask = torch.cat((embed_time_sequence_mask, sequence_mask), dim=0)  # [T+1, B, 1]

        encoded_combined_vectors = self.encoder(combined_vectors, combined_padding_mask, combined_sequence_mask)
        combined_vectors = encoded_combined_vectors + combined_vectors

        encoded_vectors = combined_vectors[1:]  # [T, B, D]
        combined_sequence_mask = combined_sequence_mask[1:]

        sequential_vectors = encoded_vectors[global_mask].contiguous()
        sequential_padding_mask = padding_mask[:, global_mask].contiguous()
        sequential_sequence_mask = sequence_mask[global_mask].contiguous()

        sequential_vectors = self.group_norm(sequential_vectors.transpose(0, 1)).transpose(0, 1)
        sequential_vectors = self.out(sequential_vectors)
        sequential_vectors = self.masking(sequential_vectors, sequential_sequence_mask)

        # The tensor is initialized to zeros
        # output_tensor = torch.zeros_like(num_vectors, batch_size, sequential_vectors.shape[-1])

        # Assign sequential_vectors to the positions specified by global_mask
        # output_tensor[global_mask] = sequential_vectors

        return sequential_vectors.transpose(0, 1), sequential_sequence_mask.transpose(0, 1)
