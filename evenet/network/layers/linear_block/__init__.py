from evenet.control.config import DotDict
from evenet.network.layers.linear_block.gru_block import GRUBlock
from evenet.network.layers.linear_block.basic_block import BasicBlock
from evenet.network.layers.linear_block.gated_block import GatedBlock
from evenet.network.layers.linear_block.resnet_block import ResNetBlock


def create_linear_block(
        options: DotDict,
        input_dim: int,
        output_dim: int,
        skip_connection: bool = False
):
    linear_block_type = options.Network.linear_block_type
    linear_block_type = linear_block_type.lower().replace("_", "").replace(" ", "")

    if linear_block_type == "resnet":
        return ResNetBlock(options, input_dim, output_dim, skip_connection)
    elif linear_block_type == 'gated':
        return GatedBlock(options, input_dim, output_dim, skip_connection)
    elif linear_block_type == 'gru':
        return GRUBlock(options, input_dim, output_dim, skip_connection)
    else:
        return BasicBlock(options, input_dim, output_dim, skip_connection)
