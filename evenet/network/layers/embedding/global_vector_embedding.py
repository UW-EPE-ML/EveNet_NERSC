from typing import Tuple

from torch import Tensor, nn

from evenet.network.layers.embedding.sequential_vector_embedding import SequentialVectorEmbedding
from evenet.control.config import DotDict

class GlobalVectorEmbedding(nn.Module):
    def __init__(self, options: DotDict, input_dim: int):
        super(GlobalVectorEmbedding, self).__init__()
        self.embedding = SequentialVectorEmbedding(options, input_dim)

    def forward(self, vectors: Tensor, time: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ Embed event-level global vectors into the same latent space as the sequential inputs.

        Parameters
        ----------
        vectors : [B, 1, I]
            Input vector data.
        time : [B, 1]
            Input time (for diffusion purpose)
        mask : [B, 1]
            Positive mask indicating that the vector is real.

        Returns
        -------
        embeddings: [1, B, D]
            Hidden activations after embedding.
        padding_mask: [B, 1]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [1, B, 1]
            Positive mask indicating jet is real.
        global_mask: [1]
            Negative mask for indicating a sequential variable or a global variable.
        """
        embeddings, padding_mask, sequence_mask, global_mask = self.embedding(vectors, time, mask)

        return embeddings, padding_mask, sequence_mask, ~global_mask
