from typing import Dict, List
from collections import OrderedDict

from torch import Tensor, nn

from evenet.control.config import DotDict
from evenet.control.event_info import EventInfo
from evenet.network.layers.branch_linear import MultiOutputBranchLinear, BranchLinear


class ClassificationDecoder(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(self, options: DotDict, event_info: EventInfo):
        super(ClassificationDecoder, self).__init__()

        # Compute training dataset statistics to fix the final weight and bias.
        # TODO: Nicer code to compute the number of classes.
        counts = {
            f"{first_level}/{second_level}": len(event_info.class_label[first_level][second_level][0]) for first_level in event_info.class_label
            for second_level in event_info.class_label[first_level]
        }
        # A unique linear decoder for each possible regression.
        networks = OrderedDict()
        for name in event_info.classification_names:
            networks[name] = BranchLinear(
                options,
                options.Network.num_classification_layers,
                counts[name]
            )

            # networks[name] = MultiOutputBranchLinear(
            #     options,
            #     options.num_classification_layers,
            #     counts[name]
            # )

        self.networks = nn.ModuleDict(networks)

    def forward(self, vectors: Dict[str, Tensor]) -> Dict[str, List[Tensor]]:
        # vectors: Dict with mapping name -> [B, D]
        # outputs: Dict with mapping name -> [B, O_name]

        return {
            key: network(vectors['/'.join(key.split('/')[:-1])])
            for key, network in self.networks.items()
        }
