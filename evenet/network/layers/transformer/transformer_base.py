from torch import Tensor, nn

from evenet.control.config import DotDict


class TransformerBase(nn.Module):
    __constants__ = ["num_layers", "hidden_dim", "num_heads", "dim_feedforward", "dropout", "transformer_activation"]

    def __init__(self, options: DotDict, num_layers: int):
        super(TransformerBase, self).__init__()

        self.num_layers = num_layers

        self.dropout = options.Training.dropout
        self.hidden_dim = options.Network.hidden_dim
        self.num_heads = options.Network.num_attention_heads
        self.transformer_activation = options.Network.transformer_activation
        self.dim_feedforward = int(round(options.Network.transformer_dim_scale * self.hidden_dim))

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        return x
