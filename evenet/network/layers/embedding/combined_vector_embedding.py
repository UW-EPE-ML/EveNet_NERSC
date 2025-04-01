from typing import Tuple, Dict

from torch import Tensor, nn

from evenet.control.config import DotDict
from evenet.control.event_info import EventInfo
from evenet.dataset.types import InputType

from evenet.network.layers.embedding.normalizer import Normalizer
from evenet.network.layers.embedding.position_embedding import PositionEmbedding
from evenet.network.layers.embedding.global_vector_embedding import GlobalVectorEmbedding
from evenet.network.layers.embedding.relative_vector_embedding import RelativeVectorEmbedding
from evenet.network.layers.embedding.sequential_vector_embedding import SequentialVectorEmbedding
from evenet.network.layers.embedding.PET_embedding import PointEdgeTransformerEmbedding


class CombinedVectorEmbedding(nn.Module):
    __constants__ = ["num_input_features"]

    def __init__(
            self,
            options: DotDict,
            event_info: EventInfo,
            input_name: str,
            input_type: str,
            mean: Dict,
            std: Dict
    ):
        super(CombinedVectorEmbedding, self).__init__()

        self.num_input_features = event_info.num_features(input_name)

        self.vector_embeddings = self.embedding_class(input_type)(options, self.num_input_features)
        self.position_embedding = PositionEmbedding(options.Network.position_embedding_dim)

        if options.Dataset.normalize_features:
            self.normalizer = Normalizer(mean[input_name], std[input_name])
        else:
            self.normalizer = nn.Identity()

    @staticmethod
    def embedding_class(embedding_type):
        if embedding_type == InputType.Sequential:
            return PointEdgeTransformerEmbedding
        elif embedding_type == InputType.Relative:
            return RelativeVectorEmbedding
        elif embedding_type == InputType.Global:
            return GlobalVectorEmbedding
        else:
            raise ValueError(f"Unknown Embedding Type: {embedding_type}")

    def forward(self, source_data: Tensor, source_time: Tensor, source_mask: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        # Normalize incoming vectors based on training statistics.
        # source_data = self.normalizer(source_data, source_mask) # Normalization step changines to evenet.network.layers.diffusion.sampler.add_perturbation

        # Embed each vector type into the same latent space.
        embeddings, padding_mask, sequence_mask, global_mask = self.vector_embeddings(
            source_data, source_time, source_mask
        )

        # Add position embedding for this input type.
        embeddings = self.position_embedding(embeddings)

        return embeddings, padding_mask, sequence_mask, global_mask
