from evenet.control.config import DotDict
from evenet.network.layers.stacked_encoder import StackedEncoder


class JetEncoder(StackedEncoder):
    def __init__(self, options: DotDict):
        super(JetEncoder, self).__init__(options, 0, options.Network.num_encoder_layers)
