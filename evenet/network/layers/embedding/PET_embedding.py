from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from evenet.control.config import DotDict
from evenet.network.layers.embedding_stack import EmbeddingStack
from evenet.network.layers.embedding.local_embedding import LocalEmbedding
from evenet.network.layers.linear_block.masking import create_masking
from evenet.network.layers.transformer.PET_transformer import PointEdgeTransformer
from evenet.network.layers.embedding.time_embedding import FourierProjection


class PointEdgeTransformerEmbedding(nn.Module):
    __constants__ = ["input_dim", "mask_sequence_vectors"]

    def __init__(self, options: DotDict, input_dim: int):
        super(PointEdgeTransformerEmbedding, self).__init__()

        self.input_dim = input_dim
        self.mask_sequence_vectors = options.Training.mask_sequence_vectors
        self.embedding_stack = EmbeddingStack(options, input_dim)

        self.fourier_projection = FourierProjection(
            num_embed=options.Network.time_fprojection_dim, projection_dim=options.Network.hidden_dim
        )
        self.masking = create_masking(options.Network.masking)
        self.time_embed = nn.Sequential(
            nn.Linear(options.Network.hidden_dim, 2 * options.Network.hidden_dim, bias=False),
            nn.GELU()
        )

        self.enable_local_embedding = options.Network.enable_local_embedding
        if self.enable_local_embedding:
            self.local_embedding = LocalEmbedding(options, input_dim)
        else:
            self.local_embedding = None

        self.PET_transformer = PointEdgeTransformer(options, options.Network.PET_num_layers)

    def forward(self, vectors: Tensor, time: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ A stack of linear blocks with each layer doubling the hidden dimension

        Parameters
        ----------
        vectors : [B, T, I]
            Input vector data.
        time : [B, 1]
            Input time (for diffusion purpose)
        mask : [B, T]
            Positive mask indicating that the jet is a real jet.

        Returns
        -------
        embeddings: [T, B, D]
            Hidden activations after embedding.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.
        global_mask: [T]
            Negative mask for indicating a sequential variable or a global variable.
        """
        batch_size, max_vectors, input_dim = vectors.shape

        # -----------------------------------------------
        # Create an negative mask for transformer layers.
        # padding_mask: [B, T]
        # -----------------------------------------------
        padding_mask = ~mask

        # -------------------------------------------------------------------------------------------------
        # Create a positive mask indicating jet is real. This is for zeroing vectors at intermediate steps.
        # Alternatively, replace it with all ones if we are not masking (basically never).
        # sequence_mask: [T, B, 1]
        # -------------------------------------------------------------------------------------------------
        sequence_mask = mask.view(batch_size, max_vectors, 1).transpose(0, 1).contiguous()
        if not self.mask_sequence_vectors:
            sequence_mask = torch.ones_like(sequence_mask)

        # ----------------------------------------------------------------------------
        # Create a negative mask indicating that all of the vectors that we embed will
        # be sequential variables and not global variables.
        # global_mask: [T]
        # ----------------------------------------------------------------------------
        global_mask = sequence_mask.new_ones((max_vectors,))

        # -------------------------------------------------------------
        # Reshape vector to have time axis first for transformer input.
        # output: [T, B, I]
        # -------------------------------------------------------------
        embeddings = vectors.transpose(0, 1).contiguous()

        # --------------------------------
        # Embed vectors into latent space.
        # output: [T, B, D]
        # --------------------------------

        encoded = self.embedding_stack(embeddings, sequence_mask)

        # -----------------------------
        # Fourier project time into latent space.
        # output: scale, shift : [T, B, D]
        # -----------------------------

        embed_time = self.fourier_projection(time)
        embed_time = embed_time.unsqueeze(1).repeat(1, encoded.size(0), 1).transpose(0, 1).contiguous()
        embed_time = self.masking(embed_time, sequence_mask)
        embed_time = self.time_embed(embed_time)
        scale, shift = torch.chunk(embed_time, 2, dim=-1)

        # --------------------------
        # Conditioning timing info via scale & shift
        # output: [T, B, D]
        # --------------------------

        encoded = encoded * (1.0 + scale) + shift

        # ----------------------------
        # Local Embedding (For Point-Edge Point Cloud)
        # output: [T, B, D]
        # ----------------------------

        if self.enable_local_embedding:
            local_embeddings = self.local_embedding(embeddings, padding_mask, sequence_mask)
            encoded = encoded + local_embeddings

        encoded = self.PET_transformer(encoded, padding_mask, sequence_mask)

        return encoded, padding_mask, sequence_mask, global_mask
