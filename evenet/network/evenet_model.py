import torch
import pickle
from evenet.control.global_config import DotDict

from evenet.network.layers.utils import RandomDrop

from evenet.network.body.normalizer import Normalizer
from evenet.network.body.embedding import GlobalVectorEmbedding, PETBody
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.classification.classification_head import ClassificationHead, RegressionHead
from evenet.network.heads.assignment.assignment_head import AssignmentHead

from torch import Tensor, nn
from typing import Dict
import re


class EvenetModel(nn.Module):
    def __init__(
            self,
            config: DotDict,
            device,
            classification: bool = True,
            regression: bool = True,
            generation: bool = False,
            assignment: bool = False,
    ):
        super().__init__()
        # Initialize the model with the given configuration
        self.options = config.options
        self.event_info = config.event_info
        # self.save_hyperparameters(self.options)
        self.include_classification = classification
        self.include_regression = regression
        self.include_generation = generation
        self.include_assignment = assignment
        self.device = device

        with open(self.options.Dataset.normalization_file, 'rb') as f:
            loaded_normalization_dict = pickle.load(f)

        self.normalization_dict = loaded_normalization_dict

        # Initialize the normalization layer
        input_normalizers_setting = dict()
        for input_name, input_type in self.event_info.input_types.items():
            input_normalizers_setting_local = {
                "log_mask": torch.tensor(
                    [feature_info.log_scale for feature_info in self.event_info.input_features[input_name]],
                    device=self.device
                ),
                "mean": loaded_normalization_dict["input_mean"][input_name].to(self.device),
                "std": loaded_normalization_dict["input_std"][input_name].to(self.device)
            }

            if input_type in input_normalizers_setting:
                for element in input_normalizers_setting[input_type]:
                    input_normalizers_setting[input_type][element] = torch.cat(
                        input_normalizers_setting[input_type][element],
                        input_normalizers_setting_local[element],
                    )
            else:
                input_normalizers_setting[input_type] = input_normalizers_setting_local
        self.sequential_normalizer = Normalizer(
            log_mask=input_normalizers_setting["SEQUENTIAL"]["log_mask"].to(self.device),
            mean=input_normalizers_setting["SEQUENTIAL"]["mean"].to(self.device),
            std=input_normalizers_setting["SEQUENTIAL"]["std"].to(self.device)
        )

        self.global_normalizer = Normalizer(
            log_mask=input_normalizers_setting["GLOBAL"]["log_mask"].to(self.device),
            mean=input_normalizers_setting["GLOBAL"]["mean"].to(self.device),
            std=input_normalizers_setting["GLOBAL"]["std"].to(self.device)
        )

        self.global_input_dim = input_normalizers_setting["GLOBAL"]["log_mask"].size()[-1]
        self.sequential_input_dim = input_normalizers_setting["SEQUENTIAL"]["log_mask"].size()[-1]

        # Initialize the embedding layers

        self.global_embedding = GlobalVectorEmbedding(
            linear_block_type=self.options.Network.linear_block_type,
            input_dim=self.global_input_dim,
            hidden_dim_scale=self.options.Network.transformer_dim_scale,
            initial_embedding_dim=self.options.Network.initial_embedding_dim,
            final_embedding_dim=self.options.Network.hidden_dim,
            normalization_type=self.options.Network.normalization,
            activation_type=self.options.Network.linear_activation,
            skip_connection=False,
            num_embedding_layers=self.options.Network.num_embedding_layers,
            dropout=self.options.Network.dropout
        )

        self.local_feature_indices = self.options.Network.local_point_index
        self.PET_body = PETBody(
            num_feat=len(self.local_feature_indices),
            num_keep=self.options.Network.num_feature_keep,
            feature_drop=self.options.Network.PET_drop_probability,
            projection_dim=self.options.Network.hidden_dim,
            local=self.options.Network.enable_local_embedding,
            K=self.options.Network.local_Krank,
            num_local=self.options.Network.num_local_layer,
            num_layers=self.options.Network.PET_num_layers,
            num_heads=self.options.Network.PET_num_heads,
            drop_probability=self.options.Network.PET_drop_probability,
            talking_head=self.options.Network.PET_talking_head,
            layer_scale=self.options.Network.PET_layer_scale,
            layer_scale_init=self.options.Network.PET_layer_scale_init,
            dropout=self.options.Network.dropout,
            mode="train"
        )

        self.object_encoder = ObjectEncoder(
            hidden_dim=self.options.Network.hidden_dim,
            position_embedding_dim=self.options.Network.position_embedding_dim,
            num_heads=self.options.Network.num_attention_heads,
            transformer_dim_scale=self.options.Network.transformer_dim_scale,
            num_linear_layers=self.options.Network.num_jet_embedding_layers,
            num_encoder_layers=self.options.Network.num_jet_encoder_layers,
            dropout=self.options.Network.dropout,
            conditioned=False
        )

        self.class_head = ClassificationHead(
            event_info=self.event_info,
            num_layers=self.options.Network.num_classification_layers,
            hidden_dim=self.options.Network.hidden_dim,
            dropout=self.options.Network.dropout,
        ) if self.include_classification else None

        self.regression_head = RegressionHead(
            event_info=self.event_info,
            means=self.normalization_dict["regression_mean"],
            stds=self.normalization_dict["regression_std"],
            num_layers=self.options.Network.num_regression_layers,
            hidden_dim=self.options.Network.hidden_dim,
            dropout=self.options.Network.dropout,
            device=self.device,
        ) if self.include_regression else None

        # Initialize the resonance particle condition

        self.num_resonance_particle_feature = self.event_info.resonance_particle_properties_mean.size(0)
        self.resonance_particle_properties = (
            nn.ParameterDict({topology_name:
                nn.Parameter(
                    self.event_info.pairing_topology[topology_name][
                        "resonance_particle_properties"].to(self.device),
                    requires_grad=False)
                for topology_name in self.event_info.pairing_topology}))

        self.resonance_particle_properties_normalizer = Normalizer(
            mean=self.event_info.resonance_particle_properties_mean.to(self.device),
            std=self.event_info.resonance_particle_properties_std.to(self.device),
            log_mask=torch.zeros_like(self.event_info.resonance_particle_properties_mean, device=self.device).bool())
        self.resonance_particle_embed = nn.Sequential(
            RandomDrop(self.options.Network.feature_drop, self.options.Network.num_feature_keep),
            nn.Linear(self.num_resonance_particle_feature, self.options.Network.hidden_dim),
            nn.GELU(),
            nn.Linear(self.options.Network.hidden_dim, self.options.Network.hidden_dim)
        )

        # Initialize the assignment head
        self.process_names = self.event_info.process_names

        self.multiprocess_assign_head = nn.ModuleDict({
            topology_name: AssignmentHead(
                split_attention=self.options.Network.split_symmetric_attention,
                hidden_dim=self.options.Network.hidden_dim,
                position_embedding_dim=self.options.Network.position_embedding_dim,
                num_heads=self.options.Network.num_attention_heads,
                transformer_dim_scale=self.options.Network.transformer_dim_scale,
                num_linear_layers=self.options.Network.num_jet_embedding_layers,
                num_encoder_layers=self.options.Network.num_jet_encoder_layers,
                num_detection_layers=self.options.Network.num_detection_layers,
                dropout=self.options.Network.dropout,
                combinatorial_scale=self.options.Network.combinatorial_scale,
                product_names=self.event_info.pairing_topology_category[topology_name]["product_particles"].names,
                product_symmetries=self.event_info.pairing_topology_category[topology_name]["product_symmetry"],
                softmax_output=True
            )
            for topology_name in self.event_info.pairing_topology_category
        })

        # Record attention head basic information
        self.permutation_indices = dict()
        self.num_targets = dict()

        for process in self.event_info.process_names:
            self.permutation_indices[process] = []
            self.num_targets[process] = []
            for event_particle_name, product_symmetry in self.event_info.product_symmetries[process].items():
                topology_name = ''.join(self.event_info.product_particles[process][event_particle_name].names)
                topology_name = f"{event_particle_name}/{topology_name}"
                topology_name = re.sub(r'\d+', '', topology_name)
                topology_category_name = self.event_info.pairing_topology[topology_name]["pairing_topology_category"]
                self.permutation_indices[process].append(
                    self.multiprocess_assign_head[topology_category_name].permutation_indices)
                self.num_targets[process].append(self.multiprocess_assign_head[topology_category_name].num_targets)

    def forward(self, x: Dict[str, Tensor], time: Tensor) -> Dict[str, Tensor]:
        """

        :param x:
            - x['x']: point cloud, shape (batch_size, num_objects, num_features)
            - x['x_mask']: Mask for point cloud, shape (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
            - x['conditions']: conditions, shape (batch_size, num_conditions)
            - x['conditions_mask']: Mask for conditions, shape (batch_size, 1)
                - 1: valid condition
                - 0: invalid condition
            - x['classification']: classification targets, shape (batch_size,)
            - x['regression']: regression targets, shape (batch_size, num_regression_targets)
            - x['regression_mask']: Mask for regression targets, shape (batch_size, num_regression_targets)
                - 1: valid regression target
                - 0: invalid regression target
            - x['num_vectors']: number of vectors in the batch, shape (batch_size,)
            - x['num_sequential_vectors']: number of sequential vectors in the batch, shape (batch_size,)
            - x['assignment_indices']: assignment indices, shape (batch_size, num_resonaces, num_targets)
            - x['assignment_indices_mask']: Mask for assignment indices, shape (batch_size, num_resonances)
                - True: valid assignment index
                - False: invalid assignment index
            - x['assignment_mask']: assignment mask, shape (batch_size, num_resonances)
                - 1: valid assignment
                - 0: invalid assignment
        """

        #############
        ##  Input  ##
        #############

        input_point_cloud = x['x']
        input_point_cloud_mask = x['x_mask'].unsqueeze(-1)
        global_conditions = x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
        global_conditions_mask = x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1)

        #########################
        ## Input normalization ##
        #########################

        input_point_cloud = self.sequential_normalizer(input_point_cloud, input_point_cloud_mask)
        global_conditions = self.global_normalizer(global_conditions, global_conditions_mask)

        #############################
        ## Central embedding (PET) ##
        #############################

        global_conditions = self.global_embedding(global_conditions, global_conditions_mask)
        local_feature = input_point_cloud[..., self.local_feature_indices]
        input_point_cloud = self.PET_body(local_feature, input_point_cloud, input_point_cloud_mask, time)

        ######################################
        ## Embedding for deterministic task ##
        ######################################

        embeddings, embedded_global_conditions, event_token = self.object_encoder(
            encoded_vectors=input_point_cloud,
            mask=input_point_cloud_mask,
            condition_vectors=global_conditions,
            condition_mask=global_conditions_mask)

        ########################################
        ## Output Head for deterministic task ##
        ########################################

        # Classification head
        classifications = None
        if self.include_classification:
            classifications = self.class_head(event_token)
        # Regression head
        regressions = None
        if self.include_regression:
            regressions = self.regression_head(event_token)

        # Assignment head
        # Create output lists for each particle in event.
        assignments = dict()
        detections = dict()

        # Pass the shared hidden state to every decoder branch

        branch_decoder_result = dict()
        for topology_name in self.event_info.pairing_topology:
            # Condition embedding for each assignment head
            topology_category_name = self.event_info.pairing_topology[topology_name]["pairing_topology_category"]
            condition_variable = self.resonance_particle_properties_normalizer(
                self.resonance_particle_properties[topology_name])
            condition_variable = self.resonance_particle_embed(condition_variable)

            (
                assignment,
                detection,
                assignment_mask,
                event_particle_vector,
                product_particle_vectors
            ) = self.multiprocess_assign_head[topology_category_name](
                embeddings, input_point_cloud_mask, embedded_global_conditions,
                global_conditions_mask, condition_variable
            )

            branch_decoder_result[topology_name] = {"assignment": assignment, "detection": detection}

        for process in self.process_names:
            assignments[process] = []
            detections[process] = []
            for event_particle_name, product_symmetry in self.event_info.product_symmetries[process].items():
                topology_name = ''.join(self.event_info.product_particles[process][event_particle_name].names)
                topology_name = f"{event_particle_name}/{topology_name}"
                topology_name = re.sub(r'\d+', '', topology_name)
                assignments[process].append(branch_decoder_result[topology_name]["assignment"].clone())
                detections[process].append(branch_decoder_result[topology_name]["detection"].clone())

        return {
            "classification": classifications,
            "regression": regressions,
            "assignments": assignments,
            "detections": detections
        }

    def shared_step(self, batch: Dict[str, Tensor], batch_size, is_training: bool = True) -> Dict[str, Tensor]:
        time = batch['x'].new_ones((batch_size,))
        output = self.forward(batch, time)
        return output
