from evenet.control.config import DotDict

from evenet.network.layers.transformer.transformer_base import TransformerBase
from evenet.network.layers.transformer.gated_transformer import GatedTransformer
from evenet.network.layers.transformer.standard_transformer import StandardTransformer
from evenet.network.layers.transformer.norm_first_transformer import NormFirstTransformer


def create_transformer(
        options: DotDict,
        num_layers: int,
):
    transformer_type = options.Network.transformer_type
    transformer_type = transformer_type.lower().replace("_", "").replace(" ", "")

    if num_layers <= 0:
        return TransformerBase(options, num_layers)

    if transformer_type == "standard":
        return StandardTransformer(options, num_layers)
    elif transformer_type == 'normfirst':
        return NormFirstTransformer(options, num_layers)
    elif transformer_type == 'gated':
        return GatedTransformer(options, num_layers)
    elif transformer_type == 'gtrxl':
        return GatedTransformer(options, num_layers)
    else:
        return TransformerBase(options, num_layers)
