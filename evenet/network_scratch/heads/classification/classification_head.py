from torch import nn, Tensor
from typing import Dict
from evenet.control.event_info import EventInfo
from evenet.network_scratch.layers.linear_block import create_linear_block
from collections import OrderedDict
import torch

import numpy as np

class BranchLinear(nn.Module):

    def __init__(
            self,
            num_layers: int,
            hidden_dim: int,
            num_outputs: int = 1,
            dropout: float = 0.0,
            batch_norm: bool = True
    ):
        super(BranchLinear, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.hidden_layers = nn.ModuleList([
            create_linear_block(
                linear_block_type="GRU",
                input_dim=self.hidden_dim,
                hidden_dim_scale=2.0,
                output_dim=self.hidden_dim,
                normalization_type="LayerNorm",
                activation_type="gelu",
                dropout=dropout,
                skip_connection=False
            ) for _ in range(self.num_layers)])

        # TODO Play around with this normalization layer
        if batch_norm:
            self.output_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.output_norm = nn.Identity()

        self.output_layer = nn.Linear(hidden_dim, num_outputs)

    def forward(self, single_vector: Tensor) -> Tensor:
        """ Produce a single classification output for a sequence of vectors.

        Parameters
        ----------
        single_vector : [B, D]
            Hidden activations after central encoder.

        Returns
        -------
        classification: [B, O]
            Probability of this particle existing in the data.
        """
        batch_size, input_dim = single_vector.shape

        # -----------------------------------------------------------------------------
        # Convert our single vector into a sequence of length 1.
        # Mostly just to re-use previous code.
        # sequence_mask: [1, B, 1]
        # single_vector: [1, B, D]
        # -----------------------------------------------------------------------------
        sequence_mask = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=single_vector.device)
        single_vector = single_vector.view(batch_size, 1, input_dim)

        # ---------------------------------------------------------------------------
        # Run through hidden layer stack first, and then take the first timestep out.
        # hidden : [B, H]
        # ----------------------------------------------------------------------------
        for layer in self.hidden_layers:
            single_vector = layer(single_vector, sequence_mask)
        hidden = single_vector.view(batch_size, self.hidden_dim)

        # ------------------------------------------------------------
        # Run through the linear layer stack and output the result
        # classification : [B, O]
        # ------------------------------------------------------------
        classification = self.output_layer(self.output_norm(hidden))

        return classification


class ClassificationHead(nn.Module):
    def __init__(
            self,
            event_info: EventInfo,
            num_layers: int,
            hidden_dim: int,
            dropout: float = 0.0,
    ):
        super(ClassificationHead, self).__init__()
        networks = OrderedDict()
        for name in event_info.class_label['EVENT']:
            num_classes = (np.array(event_info.class_label['EVENT'][name])).shape[-1]
            networks[f"classification/{name}"] = BranchLinear(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_outputs=num_classes,
                dropout=dropout,
                batch_norm=True
            )
        self.networks = nn.ModuleDict(networks)

    def forward(self, x) -> Dict[str, Tensor]:
        """
        :param x: input point cloud (batch_size, num_objects, num_features)
        :param mask: mask for point cloud (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
        :return: tensor (batch_size, num_objects, num_features)
        """

        return {
            key: network(x)
            for key, network in self.networks.items()
        }
