from typing import Tuple

import torch
from torch import Tensor, nn

from evenet.control.config import DotDict
from evenet.network.layers.transformer import create_transformer
from evenet.network.layers.linear_stack import create_linear_stack


class StackedEncoder(nn.Module):
    def __init__(
            self,
            options: DotDict,
            num_linear_layers: int,
            num_encoder_layers: int,
            conditioned: bool = False
    ):
        super(StackedEncoder, self).__init__()

        self.hidden_dim = options.Network.hidden_dim
        self.particle_vector = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        if conditioned:
            self.dense_particle = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        self.encoder = create_transformer(options, num_encoder_layers)
        self.embedding = create_linear_stack(
            options, num_linear_layers, self.hidden_dim,
            options.Network.skip_connections
        )

    def forward(self, encoded_vectors: Tensor, padding_mask: Tensor, sequence_mask: Tensor,
                cond_vector: Tensor = None) -> Tuple[Tensor, Tensor]:
        """ Apply time-independent linear layers followed by a transformer encoder.

        This is used during the branches and symmetric attention layers.

        Parameters
        ----------
        encoded_vectors: [T, B, D]
            Input sequence to predict on.
        padding_mask : [B, T]
            Negative mask for transformer input.
        sequence_mask : [T, B, 1]
            Positive mask for zeroing out padded vectors between operations.
        cond_vector   : [1, B, D]

        Returns
        -------
        output : [T, B, 1]
            New encoded vectors.
        """
        num_vectors, batch_size, hidden_dim = encoded_vectors.shape

        # -----------------------------------------------------------------------------
        # Embed vectors again into particle space
        # vectors: [T, B, D]
        # -----------------------------------------------------------------------------
        encoded_vectors = self.embedding(encoded_vectors, sequence_mask)

        # -----------------------------------------------------------------------------
        # Add a "particle vector" which will store particle level data.
        # particle_vector: [1, B, D]
        # combined_vectors: [T + 1, B, D]
        # -----------------------------------------------------------------------------
        particle_vector = self.particle_vector.expand(1, batch_size, hidden_dim)

        if cond_vector is not None:
            cond_vector_expand = cond_vector.expand(1, batch_size, hidden_dim)
            particle_vector = torch.cat((particle_vector, cond_vector_expand), dim=-1)
            particle_vector = self.dense_particle(particle_vector)

        combined_vectors = torch.cat((particle_vector, encoded_vectors), dim=0)

        # -----------------------------------------------------------------------------
        # Also modify the padding mask to indicate that the particle vector is real.
        # particle_padding_mask: [B, 1]
        # combined_padding_mask: [B, T + 1]
        # -----------------------------------------------------------------------------
        particle_padding_mask = padding_mask.new_zeros(batch_size, 1)
        combined_padding_mask = torch.cat((particle_padding_mask, padding_mask), dim=1)

        # -----------------------------------------------------------------------------
        # Also modify the sequence mask to indicate that the particle vector is real.
        # particle_sequence_mask: [1, B, 1]
        # combined_sequence_mask: [T + 1, B, 1]
        # -----------------------------------------------------------------------------
        particle_sequence_mask = sequence_mask.new_ones(1, batch_size, 1, dtype=torch.bool)
        combined_sequence_mask = torch.cat((particle_sequence_mask, sequence_mask), dim=0)

        # -----------------------------------------------------------------------------
        # Run all of the vectors through transformer encoder
        # combined_vectors: [T + 1, B, D]
        # particle_vector: [B, D]
        # encoded_vectors: [T, B, D]
        # -----------------------------------------------------------------------------
        combined_vectors = self.encoder(combined_vectors, combined_padding_mask, combined_sequence_mask)
        particle_vector, encoded_vectors = combined_vectors[0], combined_vectors[1:]

        return encoded_vectors, particle_vector
