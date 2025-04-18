import torch
import pickle
from evenet.control.global_config import DotDict

from evenet.network.layers.utils import RandomDrop

from evenet.network.body.normalizer import Normalizer
from evenet.network.body.embedding import GlobalVectorEmbedding, PETBody
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.classification.classification_head import ClassificationHead, RegressionHead
from evenet.network.heads.assignment.assignment_head import SharedAssignmentHead
from evenet.network.heads.generation.generation_head import GlobalCondGenerationHead, EventGenerationHead
from evenet.network.layers.debug_layer import PointCloudTransformer
from evenet.utilities.group_theory import complete_indices

from evenet.utilities.diffusion_sampler import add_noise
from evenet.utilities.tool import gather_index
from torch import Tensor, nn
from typing import Dict, Optional, Any
import re


class EveNetModel(nn.Module):
    def __init__(
            self,
            config: DotDict,
            device,
            classification: bool = True,
            regression: bool = False,
            generation: bool = False,
            neutrino_generation: bool = False,
            assignment: bool = False,
            normalization_dict: dict = None,
    ):
        super().__init__()
        # # Initialize the model with the given configuration
        self.options = config.options
        self.network_cfg = config.network
        self.event_info = config.event_info
        # # self.save_hyperparameters(self.options)
        self.include_classification = classification
        self.include_regression = regression
        self.include_generation = generation
        self.include_neutrino_generation = neutrino_generation
        self.include_assignment = assignment
        self.device = device

        self.normalization_dict = normalization_dict

        # Initialize the normalization layer
        input_normalizers_setting = dict()
        for input_name, input_type in self.event_info.input_types.items():
            input_normalizers_setting_local = {
                "norm_mask": torch.tensor(
                    [feature_info.normalize for feature_info in self.event_info.input_features[input_name]],
                    device=self.device
                ),
                "mean": normalization_dict["input_mean"][input_name].to(self.device),
                "std": normalization_dict["input_std"][input_name].to(self.device)
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
            norm_mask=input_normalizers_setting["SEQUENTIAL"]["norm_mask"].to(self.device),
            mean=input_normalizers_setting["SEQUENTIAL"]["mean"].to(self.device),
            std=input_normalizers_setting["SEQUENTIAL"]["std"].to(self.device),
            inv_cdf_index=self.event_info.sequential_inv_cdf_index
        )

        global_normalizer_info = input_normalizers_setting.get(
            "GLOBAL",
            {
                "norm_mask": torch.ones(1, dtype=torch.bool).to(self.device),
                "mean": torch.zeros(1).to(self.device),
                "std": torch.ones(1).to(self.device)
            }
        )
        self.global_normalizer = Normalizer(
            norm_mask=global_normalizer_info["norm_mask"].to(self.device),
            mean=global_normalizer_info["mean"].to(self.device),
            std=global_normalizer_info["std"].to(self.device),
        )

        self.num_point_cloud_normalizer = Normalizer(
            mean=normalization_dict["input_num_mean"]["Source"].unsqueeze(-1).to(self.device),
            std=normalization_dict["input_num_std"]["Source"].unsqueeze(-1).to(self.device),
            norm_mask=torch.tensor([1], device=self.device, dtype=torch.bool)
        )

        self.global_input_dim = global_normalizer_info["norm_mask"].size()[-1]
        self.sequential_input_dim = input_normalizers_setting["SEQUENTIAL"]["norm_mask"].size()[-1]
        self.local_feature_indices = self.network_cfg.Body.PET.local_point_index

        # [1] Body
        global_embedding_cfg = self.network_cfg.Body.GlobalEmbedding
        self.GlobalEmbedding = GlobalVectorEmbedding(
            linear_block_type=global_embedding_cfg.linear_block_type,
            input_dim=self.global_input_dim,
            hidden_dim_scale=global_embedding_cfg.transformer_dim_scale,
            initial_embedding_dim=global_embedding_cfg.initial_embedding_dim,
            final_embedding_dim=global_embedding_cfg.hidden_dim,
            normalization_type=global_embedding_cfg.normalization,
            activation_type=global_embedding_cfg.linear_activation,
            skip_connection=False,
            num_embedding_layers=global_embedding_cfg.num_embedding_layers,
            dropout=global_embedding_cfg.dropout
        )

        # tihsu: debugging
        # self.global_embedding_debug = nn.Sequential(
        #     nn.Linear(self.network_cfg.hidden_dim, self.network_cfg.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.network_cfg.hidden_dim, self.network_cfg.hidden_dim),
        #     nn.ReLU()
        #     )
        # self.point_cloud_transformer_debug = PointCloudTransformer(
        #     point_dim=self.network_cfg.hidden_dim,
        #     embed_dim=self.network_cfg.hidden_dim,
        #     ff_dim=self.network_cfg.hidden_dim
        # )

        # self.debug_classifier = nn.Linear(self.network_cfg.hidden_dim, self.event_info.num_classes["signal"])
        # [1] Body
        pet_config = self.network_cfg.Body.PET
        self.PET = PETBody(
            num_feat=self.sequential_input_dim,
            num_keep=pet_config.num_feature_keep,
            feature_drop=pet_config.feature_drop,
            projection_dim=pet_config.hidden_dim,
            local=pet_config.enable_local_embedding,
            K=pet_config.local_Krank,
            num_local=pet_config.num_local_layer,
            num_layers=pet_config.num_layers,
            num_heads=pet_config.num_heads,
            drop_probability=pet_config.drop_probability,
            talking_head=pet_config.talking_head,
            layer_scale=pet_config.layer_scale,
            layer_scale_init=pet_config.layer_scale_init,
            dropout=pet_config.dropout,
            mode=pet_config.mode,
        )

        # [2] Classification + Regression + Assignment Body
        obj_encoder_cfg = self.network_cfg.Body.ObjectEncoder
        self.ObjectEncoder = ObjectEncoder(
            input_dim=pet_config.hidden_dim,
            hidden_dim=obj_encoder_cfg.hidden_dim,
            output_dim=obj_encoder_cfg.hidden_dim,
            position_embedding_dim=obj_encoder_cfg.position_embedding_dim,
            num_heads=obj_encoder_cfg.num_attention_heads,
            transformer_dim_scale=obj_encoder_cfg.transformer_dim_scale,
            num_linear_layers=obj_encoder_cfg.num_embedding_layers,
            num_encoder_layers=obj_encoder_cfg.num_encoder_layers,
            dropout=obj_encoder_cfg.dropout,
            conditioned=False,
            skip_connection=obj_encoder_cfg.skip_connection,
            encoder_skip_connection=obj_encoder_cfg.encoder_skip_connection,
        )

        # [3] Classification Head
        if self.include_classification:
            cls_cfg = self.network_cfg.Classification
            self.Classification = ClassificationHead(
                input_dim=obj_encoder_cfg.hidden_dim,
                class_label=self.event_info.class_label.get("EVENT", None),
                event_num_classes=self.event_info.num_classes,
                num_layers=cls_cfg.num_classification_layers,
                hidden_dim=cls_cfg.hidden_dim,
                dropout=cls_cfg.dropout,
            )
        # [4] Regression Head
        if self.include_regression:
            reg_cfg = self.network_cfg.Regression
            self.Regression = RegressionHead(
                input_dim=obj_encoder_cfg.hidden_dim,
                regressions_target=self.event_info.regressions,
                regression_names=self.event_info.regression_names,
                means=self.normalization_dict["regression_mean"],
                stds=self.normalization_dict["regression_std"],
                num_layers=reg_cfg.num_regression_layers,
                hidden_dim=reg_cfg.hidden_dim,
                dropout=reg_cfg.dropout,
                device=self.device,
            )

        if self.include_assignment:
            # [5] Assignment Head
            self.Assignment = SharedAssignmentHead(
                resonance_particle_properties_mean=self.event_info.resonance_particle_properties_mean,
                resonance_particle_properties_std=self.event_info.resonance_particle_properties_std,
                pairing_topology=self.event_info.pairing_topology,
                process_names=self.event_info.process_names,
                pairing_topology_category=self.event_info.pairing_topology_category,
                event_particles=self.event_info.event_particles,
                event_permutation=self.event_info.event_permutations,
                product_particles=self.event_info.product_particles,
                product_symmetries=self.event_info.product_symmetries,
                feature_drop=self.network_cfg.Assignment.feature_drop,
                num_feature_keep=self.network_cfg.Assignment.num_feature_keep,
                input_dim=obj_encoder_cfg.hidden_dim,
                split_attention=self.network_cfg.Assignment.split_symmetric_attention,
                hidden_dim=self.network_cfg.Assignment.hidden_dim,
                position_embedding_dim=self.network_cfg.Assignment.position_embedding_dim,
                num_attention_heads=self.network_cfg.Assignment.num_attention_heads,
                transformer_dim_scale=self.network_cfg.Assignment.transformer_dim_scale,
                num_linear_layers=self.network_cfg.Assignment.num_linear_layers,
                num_encoder_layers=self.network_cfg.Assignment.num_encoder_layers,
                num_jet_embedding_layers=self.network_cfg.Assignment.num_jet_embedding_layers,
                num_jet_encoder_layers=self.network_cfg.Assignment.num_jet_encoder_layers,
                num_max_event_particles=self.event_info.max_event_particles,
                num_detection_layers=self.network_cfg.Assignment.num_detection_layers,
                dropout=self.network_cfg.Assignment.dropout,
                combinatorial_scale=self.network_cfg.Assignment.combinatorial_scale,
                encode_event_token=self.network_cfg.Assignment.encode_event_token,
                activation=self.network_cfg.Assignment.activation,
                skip_connection=self.network_cfg.Assignment.skip_connection,
                encoder_skip_connection=self.network_cfg.Assignment.encoder_skip_connection,
                device=self.device
            )

        # [6] Generation Head
        if self.include_generation:
            # [6-1] Global Generation Head
            self.GlobalGeneration = GlobalCondGenerationHead(
                num_layer=self.network_cfg.GlobalGeneration.num_layers,
                num_resnet_layer=self.network_cfg.GlobalGeneration.num_resnet_layers,
                input_dim=1,  # Only target on the number of point_cloud
                hidden_dim=self.network_cfg.GlobalGeneration.hidden_dim,
                output_dim=1,
                input_cond_indices=self.event_info.generation_condition_indices,
                num_classes=self.event_info.num_classes_total,
                resnet_dim=self.network_cfg.GlobalGeneration.resnet_dim,
                layer_scale_init=self.network_cfg.GlobalGeneration.layer_scale_init,
                feature_drop_for_stochastic_depth=self.network_cfg.GlobalGeneration.feature_drop_for_stochastic_depth,
                activation=self.network_cfg.GlobalGeneration.activation,
                dropout=self.network_cfg.GlobalGeneration.dropout
            )
            # [6-2] Event Generation Head
            self.EventGeneration = EventGenerationHead(
                input_dim=pet_config.hidden_dim,
                projection_dim=self.network_cfg.EventGeneration.hidden_dim,
                num_global_cond=global_embedding_cfg.hidden_dim,
                num_classes=self.event_info.num_classes_total,
                output_dim=self.sequential_input_dim,
                num_layers=self.network_cfg.EventGeneration.num_layers,
                num_heads=self.network_cfg.EventGeneration.num_heads,
                dropout=self.network_cfg.EventGeneration.dropout,
                layer_scale=self.network_cfg.EventGeneration.layer_scale,
                layer_scale_init=self.network_cfg.EventGeneration.layer_scale_init,
                drop_probability=self.network_cfg.EventGeneration.drop_probability,
                feature_drop=self.network_cfg.EventGeneration.feature_drop
            )
        self.schedule_flags = [
            (self.include_classification or self.include_assignment or self.include_regression, "deterministic"),
            (self.include_generation, "generation"),
            (self.include_neutrino_generation, "neutrino_generation"),
        ]

    def forward(self, x: Dict[str, Tensor], time: Tensor) -> dict[str, dict[Any, Any] | Any]:
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

            - x['x_invisible']: invisible point cloud, shape (batch_size, num_objects, num_features)
            - x['x_invisible_mask']: Mask for invisible point cloud, shape (batch_size, num_objects)
        """

        #############
        ##  Input  ##
        #############

        input_point_cloud = x['x']
        input_point_cloud_mask = x['x_mask'].unsqueeze(-1)
        global_conditions = x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
        global_conditions_mask = x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1, 1)

        class_label = x['classification'].unsqueeze(-1) if 'classification' in x else torch.zeros_like(
            x['conditions_mask']).long()  # (batch_size, 1)
        num_point_cloud = x['num_sequential_vectors'].unsqueeze(-1)  # (batch_size, 1)

        invisible_point_cloud = x['x_invisible'] if 'x_invisible' in x else torch.zeros_like(
            input_point_cloud[:, [0], :])
        invisible_point_cloud_mask = x['x_invisible_mask'].unsqueeze(
            -1) if 'x_invisible_mask' in x else torch.zeros_like(input_point_cloud_mask[:, [0], :]).bool()

        #######################################################
        ##  Produce visible + invisible point cloud masking  ##
        #######################################################

        # Requirement: input_point_cloud and invisible_point_cloud should have the same features

        # Create attention mask
        n_vis = input_point_cloud_mask.shape[1]
        n_invis = invisible_point_cloud_mask.shape[1]
        is_invisible_query = torch.cat([
            torch.zeros(n_vis, dtype=torch.bool, device=self.device),
            torch.ones(n_invis, dtype=torch.bool, device=self.device)
        ], dim=0)  # (L,)
        # Rule: visible query (False) cannot attend to invisible key (True)
        invisible_attn_mask = (~is_invisible_query[:, None]) & is_invisible_query[None, :]  # (L, L) , Q->K

        #########################
        ## Input normalization ##
        #########################

        input_point_cloud = self.sequential_normalizer(
            x=input_point_cloud,
            mask=input_point_cloud_mask,
        )

        invisible_point_cloud = self.sequential_normalizer(
            x=invisible_point_cloud,
            mask=invisible_point_cloud_mask
        )

        global_conditions = self.global_normalizer(
            x=global_conditions,
            mask=global_conditions_mask
        )
        num_point_cloud = self.num_point_cloud_normalizer(
            x=num_point_cloud,
            mask=None
        )

        ###########################
        ## Global Generator Head ##
        ###########################

        generations = dict()
        if self.include_generation:
            num_point_cloud_noised, truth_num_point_cloud_vector = add_noise(num_point_cloud, time)
            predict_num_point_cloud_vector = self.GlobalGeneration(
                x=num_point_cloud_noised,
                time=time,
                global_cond=global_conditions,
                label=class_label
            )
            generations["num_point_cloud"] = {
                "vector": predict_num_point_cloud_vector,
                "truth": truth_num_point_cloud_vector
            }

        outputs = dict()
        for flag, schedule_name in self.schedule_flags:
            if not flag:
                continue

            ####################
            ##  Inject noise  ##
            ####################

            if schedule_name == "deterministic":
                full_input_point_cloud = input_point_cloud.contiguous()
                full_input_point_cloud_mask = input_point_cloud_mask.contiguous()
                full_attn_mask = None
                full_time = torch.zeros_like(time)
                time_masking = torch.zeros_like(full_input_point_cloud_mask).float()
            elif schedule_name == "generation":
                input_point_cloud_noised, truth_input_point_cloud_vector = add_noise(input_point_cloud, time)
                full_input_point_cloud = input_point_cloud_noised.contiguous()
                full_input_point_cloud_mask = input_point_cloud_mask.contiguous()
                full_attn_mask = None
                full_time = time.contiguous()
                time_masking = full_input_point_cloud_mask.float()
            else:
                invisible_point_cloud_noised, truth_invisible_point_cloud_vector = add_noise(invisible_point_cloud,
                                                                                             time)
                full_input_point_cloud = torch.cat([input_point_cloud, invisible_point_cloud_noised], dim=1)
                full_input_point_cloud_mask = torch.cat([input_point_cloud_mask, invisible_point_cloud_mask], dim=1)
                full_attn_mask = invisible_attn_mask.contiguous()
                full_time = time.contiguous()
                time_masking = torch.cat([torch.zeros_like(input_point_cloud_mask), invisible_point_cloud_mask],
                                         dim=1).float()

            #############################
            ## Central embedding (PET) ##
            #############################

            full_global_conditions = self.GlobalEmbedding(
                x=global_conditions,
                mask=global_conditions_mask
            )

            local_points = full_input_point_cloud[..., self.local_feature_indices]
            full_input_point_cloud = self.PET(
                input_features=full_input_point_cloud,
                input_points=local_points,
                mask=full_input_point_cloud_mask,
                attn_mask=full_attn_mask,
                time=full_time,
                time_masking=time_masking
            )

            if schedule_name == "deterministic" or schedule_name == "generation":

                ######################################
                ## Embedding for deterministic task ##
                ######################################

                embeddings, embedded_global_conditions, event_token = self.ObjectEncoder(
                    encoded_vectors=full_input_point_cloud,
                    mask=full_input_point_cloud_mask,
                    condition_vectors=full_global_conditions,
                    condition_mask=global_conditions_mask
                )

                ########################################
                ## Output Head for deterministic task ##
                ########################################

                # Assignment head
                # Create output lists for each particle in event.
                assignments = dict()
                detections = dict()
                if self.include_assignment:
                    assignments, detections, event_token = self.Assignment(
                        x=embeddings,
                        x_mask=full_input_point_cloud_mask,
                        global_condition=embedded_global_conditions,
                        global_condition_mask=global_conditions_mask,
                        event_token=event_token,
                        return_type="process_base"
                    )

                # Classification head
                classifications = None
                if self.include_classification:
                    classifications = self.Classification(event_token)
                    # classifications = {"signal": self.debug_classifier(event_token_debug)}

                # Regression head
                regressions = None
                if self.include_regression:
                    regressions = self.Regression(event_token)

                outputs[schedule_name] = {
                    "classification": classifications,
                    "regression": regressions,
                    "assignments": assignments,
                    "detections": detections
                }

            if schedule_name == "neutrino_generation" or schedule_name == "generation":

                #######################################
                ##  Output Head For Diffusion Model  ##
                #######################################

                if self.include_generation:
                    pred_point_cloud_vector = self.EventGeneration(
                        x=full_input_point_cloud,
                        x_mask=full_input_point_cloud_mask,
                        global_cond=full_global_conditions,
                        global_cond_mask=global_conditions_mask,
                        num_x=num_point_cloud,
                        time=full_time,
                        label=class_label,
                        attn_mask=full_attn_mask,
                        time_masking=time_masking
                    )

                    if schedule_name == "neutrino_generation":
                        generations["neutrino"] = {
                            "vector": pred_point_cloud_vector[:, is_invisible_query, :],
                            "truth": truth_invisible_point_cloud_vector
                        }
                    else:
                        generations["point_cloud"] = {
                            "vector": pred_point_cloud_vector,
                            "truth": truth_input_point_cloud_vector
                        }

        return {
            "classification": outputs.get("deterministic", {}).get("classification", None),
            "regression": outputs.get("deterministic", {}).get("regression", None),
            "assignments": outputs.get("deterministic", {}).get("assignments", None),
            "detections": outputs.get("deterministic", {}).get("detections", None),
            "classification-noised": outputs.get("generation", {}).get("classification", None),
            "regression-noised": outputs.get("generation", {}).get("regression", None),
            "generations": generations
        }

    def predict_diffusion_vector(
            self, noise_x: Tensor, cond_x: Dict[str, Tensor], time: Tensor, mode: str,
            noise_mask: Optional[Tensor] = None
    ) -> Tensor:

        """
        Predict the number of point clouds in the batch.
        """

        batch_size = noise_x.shape[0]

        if mode == "global":
            """
            Predict the number of point clouds diffusion vector in the batchs.
            noise_x: (batch_size, 1)
            """
            global_conditions = cond_x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
            global_conditions_mask = cond_x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1)
            class_label = cond_x['classification'].unsqueeze(-1)  # (batch_size, 1)
            global_conditions = self.global_normalizer(
                x=global_conditions,
                mask=global_conditions_mask
            )
            predict_num_point_cloud_vector = self.GlobalGeneration(
                x=noise_x,
                time=time,
                global_cond=global_conditions,
                label=class_label
            )
            return predict_num_point_cloud_vector

        elif mode == "event":
            global_conditions = cond_x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
            global_conditions_mask = cond_x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1)
            class_label = cond_x['classification'].unsqueeze(-1)  # (batch_size, 1)
            num_point_cloud = cond_x['num_sequential_vectors'].unsqueeze(-1)  # (batch_size, 1)

            global_conditions = self.global_normalizer(
                x=global_conditions,
                mask=global_conditions_mask
            )
            num_point_cloud = self.num_point_cloud_normalizer(
                x=num_point_cloud,
                mask=None
            )
            global_conditions = self.GlobalEmbedding(
                x=global_conditions,
                mask=global_conditions_mask
            )

            local_points = noise_x[..., self.local_feature_indices]
            input_point_cloud = self.PET(
                input_features=noise_x,
                input_points=local_points,
                mask=noise_mask,
                time=time
            )
            pred_point_cloud_vector = self.EventGeneration(
                x=input_point_cloud,
                x_mask=noise_mask,
                global_cond=global_conditions,
                global_cond_mask=global_conditions_mask,
                num_x=num_point_cloud,
                time=time,
                label=class_label
            )
            return pred_point_cloud_vector

        elif mode == "neutrino":
            global_conditions = cond_x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
            global_conditions_mask = cond_x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1, 1)
            class_label = cond_x['classification'].unsqueeze(-1)  # (batch_size, 1)
            num_point_cloud = cond_x['num_sequential_vectors'].unsqueeze(-1)  # (batch_size, 1)
            input_point_cloud = cond_x['x']
            input_point_cloud_mask = cond_x['x_mask'].unsqueeze(-1)

            input_point_cloud = self.sequential_normalizer(
                x=input_point_cloud,
                mask=input_point_cloud_mask,
            )

            global_conditions = self.global_normalizer(
                x=global_conditions,
                mask=global_conditions_mask
            )

            num_point_cloud = self.num_point_cloud_normalizer(
                x=num_point_cloud,
                mask=None
            )

            invisible_point_cloud_noised = noise_x
            invisible_point_cloud_mask = noise_mask

            # Create attention mask
            n_vis = input_point_cloud_mask.shape[1]
            n_invis = invisible_point_cloud_mask.shape[1]
            is_invisible_query = torch.cat([
                torch.zeros(n_vis, dtype=torch.bool, device=self.device),
                torch.ones(n_invis, dtype=torch.bool, device=self.device)
            ], dim=0)  # (L,)
            # Rule: visible query (False) cannot attend to invisible key (True)
            invisible_attn_mask = (~is_invisible_query[:, None]) & is_invisible_query[None, :]  # (L, L) , Q->K

            full_input_point_cloud = torch.cat([input_point_cloud, invisible_point_cloud_noised], dim=1)
            full_input_point_cloud_mask = torch.cat([input_point_cloud_mask, invisible_point_cloud_mask], dim=1)
            full_attn_mask = invisible_attn_mask.contiguous()
            full_time = time.contiguous()
            time_masking = torch.cat([torch.zeros_like(input_point_cloud_mask), invisible_point_cloud_mask],
                                     dim=1).float()
            full_global_conditions = self.GlobalEmbedding(
                x=global_conditions,
                mask=global_conditions_mask
            )

            local_points = full_input_point_cloud[..., self.local_feature_indices]
            full_input_point_cloud = self.PET(
                input_features=full_input_point_cloud,
                input_points=local_points,
                mask=full_input_point_cloud_mask,
                attn_mask=full_attn_mask,
                time=full_time,
                time_masking=time_masking
            )

            pred_point_cloud_vector = self.EventGeneration(
                x=full_input_point_cloud,
                x_mask=full_input_point_cloud_mask,
                global_cond=full_global_conditions,
                global_cond_mask=global_conditions_mask,
                num_x=num_point_cloud,
                time=full_time,
                label=class_label,
                attn_mask=full_attn_mask,
                time_masking=time_masking
            )
            return pred_point_cloud_vector[:, is_invisible_query, :]
        return None

    def shared_step(self, batch: Dict[str, Tensor], batch_size, is_training: bool = True) -> dict:
        time = torch.rand((batch_size,), device=batch['x'].device, dtype=batch['x'].dtype)
        output = self.forward(batch, time)
        return output

    def freeze_module(self, logical_name: str, cfg: dict):
        """
        Freeze parameters of a head using main_modules_name lookup and freeze config.

        Parameters
        ----------
        logical_name : str
            Logical name used in main_modules_name dict (e.g., "classification_head").
        cfg : dict
            Configuration dict under that head (from config.Classification etc).
        """
        head_module = getattr(self, logical_name, None)
        if head_module is None:
            print(f"[Warning] Attribute '{logical_name}' not found")
            return

        freeze_type = cfg.get("type", "none")
        components = cfg.get("partial_freeze_components", [])

        if freeze_type == "none":
            return

        elif freeze_type == "full":
            for param in head_module.parameters():
                param.requires_grad = False

        elif freeze_type == "partial":
            for name, module in head_module.named_modules():
                if name in components:
                    for param in module.parameters():
                        param.requires_grad = False

        elif freeze_type == "random":
            import random
            freeze_fraction = cfg.get("freeze_fraction", 0.5)
            all_params = list(head_module.parameters())
            num_to_freeze = int(len(all_params) * freeze_fraction)
            to_freeze = random.sample(all_params, num_to_freeze)
            for param in to_freeze:
                param.requires_grad = False

        else:
            raise ValueError(f"Unsupported freeze type: {freeze_type}")
