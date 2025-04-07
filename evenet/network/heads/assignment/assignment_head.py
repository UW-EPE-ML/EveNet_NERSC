from typing import Tuple, List
from opt_einsum import contract_expression

import torch
from torch import nn, Tensor

from evenet.dataset.types import Symmetries
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.assignment.symmetric_attention import SymmetricAttentionSplit, SymmetricAttentionFull
from evenet.network.heads.classification.classification_head import BranchLinear
from evenet.utilities.masked_softmax_no_gradient import masked_log_softmax


class AssignmentHead(nn.Module):
    WEIGHTS_INDEX_NAMES = "ijklmn"
    DEFAULT_JET_COUNT = 16

    def __init__(
            self,
            split_attention: bool,
            hidden_dim: int,
            position_embedding_dim: int,
            num_heads: int,
            transformer_dim_scale: float,
            num_linear_layers: int,
            num_encoder_layers: int,
            num_detection_layers: int,
            dropout: float,
            combinatorial_scale: float,
            product_names: List[str],
            product_symmetries: Symmetries,
            detection_output_dim: int = 1,
            softmax_output: bool = True
    ):
        super(AssignmentHead, self).__init__()

        self.degree = product_symmetries.degree
        self.product_names = product_names
        self.softmax_output = softmax_output

        self.combinatorial_scale = combinatorial_scale

        self.encoder = ObjectEncoder(
            hidden_dim=hidden_dim,
            position_embedding_dim=position_embedding_dim,
            num_heads=num_heads,
            transformer_dim_scale=transformer_dim_scale,
            num_linear_layers=num_linear_layers,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            conditioned=True)

        attention_layer = SymmetricAttentionSplit if split_attention else SymmetricAttentionFull
        self.attention = attention_layer(
            hidden_dim=hidden_dim,
            position_embedding_dim=position_embedding_dim,
            num_heads=num_heads,
            transformer_dim_scale=transformer_dim_scale,
            num_linear_layers=num_linear_layers,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            degree=self.degree,
            permutation_indices=product_symmetries.permutations
        )

        self.detection_classifier = BranchLinear(
            num_layers=num_detection_layers,
            hidden_dim=hidden_dim,
            num_outputs=detection_output_dim,
            dropout=dropout,
            batch_norm=True
        )

        self.num_targets = len(self.attention.permutation_group)
        self.permutation_indices = self.attention.permutation_indices

        self.padding_mask_operation = self.create_padding_mask_operation()
        self.diagonal_mask_operation = self.create_diagonal_mask_operation()
        self.diagonal_mask = {}

    def create_padding_mask_operation(self):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))
        expression = f"{operands}->b{weights_index_names}"
        return expression

    def create_diagonal_mask_operation(self):
            weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]
            operands = ','.join(map(lambda x: 'b' + x, weights_index_names))
            expression = f"{operands}->{weights_index_names}"
            return expression

    def create_output_mask(self, output: Tensor, mask: Tensor) -> Tensor:

        num_jets = output.shape[1] # TODO: Double Check

        # batch_sequence_mask: [B, T, 1] Positive mask indicating jet is real.
        batch_sequence_mask = mask.contiguous()

        # =========================================================================================
        # Padding mask
        # =========================================================================================
        padding_mask_operands = [batch_sequence_mask.squeeze(-1) * 1] * self.degree
        padding_mask = torch.einsum(self.padding_mask_operation, *padding_mask_operands)
        padding_mask = padding_mask.bool()

        # =========================================================================================
        # Diagonal mask
        # =========================================================================================
        try:
            diagonal_mask = self.diagonal_mask[(num_jets, output.device)]
        except KeyError:
            identity = 1 - torch.eye(num_jets)
            identity = identity.type_as(output)

            diagonal_mask_operands = [identity * 1] * self.degree
            diagonal_mask = torch.einsum(self.diagonal_mask_operation, *diagonal_mask_operands)
            diagonal_mask = diagonal_mask.unsqueeze(0) < (num_jets + 1 - self.degree)
            self.diagonal_mask[(num_jets, output.device)] = diagonal_mask

        return (padding_mask & diagonal_mask).bool()

    def forward(
            self,
            point_cloud: Tensor,
            point_cloud_mask: Tensor,
            global_condition: Tensor,
            global_condition_mask: Tensor,
            cond_vector: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """ Create a distribution over jets for a given particle and a probability of its existence.

        Parameters
        ----------
        point_clould: Tensor, shape (batch_size, num_vectors, hidden_dim)
        point_cloud_mask: Tensor, shape (batch_size, num_vectors, 1)
        global_condition: Tensor, shape (batch_size, 1, hidden_dim)
        global_condition_mask: Tensor, shape (batch_size, 1, 1)
        cond_vector: Tensor, shape (batch_size, 1, hidden_dim)

        Returns
        -------
        selection : [TS, TS, ...]
            Distribution over sequential vectors for the target vectors.
        classification: [B]
            Probability of this particle existing in the data.
        """

        # ------------------------------------------------------
        # Apply the branch's independent encoder to each vector.
        # particle_vectors : (batch_size, num_vectors, hidden_dim)
        # ------------------------------------------------------

        encoded_vectors, encoded_global_cond, particle_vector = self.encoder(
            point_cloud, point_cloud_mask,
            global_condition, global_condition_mask,
            cond_vector
        )

        # -----------------------------------------------
        # Run the encoded vectors through the classifier.
        # detection: [B, 1]
        # -----------------------------------------------
        detection = self.detection_classifier(particle_vector).squeeze(-1)

        # --------------------------------------------------------
        # Extract sequential vectors only for the assignment step.
        # sequential_particle_vectors : [TS, B, D]
        # sequential_padding_mask : [B, TS]
        # sequential_sequence_mask : [TS, B, 1]
        # --------------------------------------------------------
        sequential_particle_vectors = encoded_vectors.contiguous()
        sequential_padding_mask = ~(point_cloud_mask.squeeze(-1)).bool().contiguous()
        sequential_sequence_mask = point_cloud_mask.contiguous()

        # --------------------------------------------------------------------
        # Create the vector distribution logits and the correctly shaped mask.
        # assignment : [TS, TS, ...]
        # assignment_mask : [TS, TS, ...]
        # --------------------------------------------------------------------
        assignment, daughter_vectors = self.attention(
            x=sequential_particle_vectors,
            x_mask=sequential_sequence_mask,
            condition=global_condition,
            condition_mask=global_condition_mask
        )

        assignment_mask = self.create_output_mask(assignment, sequential_sequence_mask)

        # ---------------------------------------------------------------------------
        # Need to reshape output to make softmax-calculation easier.
        # We transform the mask and output into a flat representation.
        # Afterwards, we apply a masked log-softmax to create the final distribution.
        # output : [TS, TS, ...]
        # mask : [TS, TS, ...]
        # ---------------------------------------------------------------------------
        if self.softmax_output:
            original_shape = assignment.shape
            batch_size = original_shape[0]

            assignment = assignment.reshape(batch_size, -1)
            assignment_mask = assignment_mask.reshape(batch_size, -1)

            assignment = masked_log_softmax(assignment, assignment_mask)
            assignment = assignment.view(*original_shape)

            # mask = mask.view(*original_shape)
            # offset = torch.log(mask.sum((1, 2, 3), keepdims=True).float()) * self.combinatorial_scale
            # output = output + offset

        return assignment, detection, assignment_mask, particle_vector, daughter_vectors
