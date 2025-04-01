from typing import List, Tuple, Dict

import torch
from torch import nn, Tensor

from evenet.control.config import DotDict
from evenet.control.event_info import EventInfo
from evenet.network.layers.linear_block import create_linear_block
from evenet.network.layers.embedding.combined_vector_embedding import CombinedVectorEmbedding


class MultiInputVectorEmbedding(nn.Module):
    def __init__(self, options: DotDict, event_info: EventInfo, mean: Dict, std: Dict):
        super(MultiInputVectorEmbedding, self).__init__()

        # Primary embedding blocks to convert each input type into an identically shaped vector.
        self.vector_embedding_layers = nn.ModuleList([
            CombinedVectorEmbedding(options, event_info, input_name, input_type, mean=mean, std=std)
            for input_name, input_type in event_info.input_types.items()
        ])

        # A final embedding layer to convert the position encoded vectors into a unified vector space.
        self.final_embedding_layer = create_linear_block(
            options,
            options.Network.position_embedding_dim + options.Network.hidden_dim,
            options.Network.hidden_dim,
            options.Network.skip_connections
        )

    def forward(self, sources: List[Tuple[Tensor, Tensor]], source_time: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        """

        Parameters
        ----------
        sources: List[[B, T_i, D_i]]
            A list containing each input source vectors in batch-first form.

        Returns
        -------
        embeddings: [T, B, D]
            Complete embeddings groups together in order of their inputs and in the proper latent dimension.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.
        global_mask: [T]
            Negative mask for indicating a sequential variable or a global variable.
        """
        embeddings = []
        padding_masks = []
        sequence_masks = []
        global_masks = []

        for input_index, vector_embedding_layer in enumerate(self.vector_embedding_layers):
            source_data, source_mask = sources[input_index]

            # Embed each vector type into the same latent space.
            current_embeddings = vector_embedding_layer(source_data, source_time, source_mask)

            # Accumulate all vectors into a single sequence.
            embeddings.append(current_embeddings[0])
            padding_masks.append(current_embeddings[1])
            sequence_masks.append(current_embeddings[2])
            global_masks.append(current_embeddings[3])

        embeddings = torch.cat(embeddings, dim=0)
        padding_masks = torch.cat(padding_masks, dim=1)
        sequence_masks = torch.cat(sequence_masks, dim=0)
        global_masks = torch.cat(global_masks, dim=0)

        embeddings = self.final_embedding_layer(embeddings, sequence_masks)

        return embeddings, padding_masks, sequence_masks, global_masks
