import torch
import pickle
from evenet.control.config import DotDict

from evenet.network_scratch.body.normalizer import normalizer
from evenet.network_scratch.body.embedding import GlobalVectorEmbedding, PETBody, CombinedEmbedding
from evenet.network_scratch.body.object_encoder import ObjectEncoder

from evenet.network_scratch.heads.classification.classification_head import ClassificationHead
from torch import Tensor, nn
from typing import Dict


class EvenetModel(nn.Module):
    def __init__(self, config: DotDict):
        super().__init__()
        # Initialize the model with the given configuration
        self.options = config.options
        self.event_info = config.event_info
        # self.save_hyperparameters(self.options)

        with open(self.options.Dataset.normalization_file, 'rb') as f:
            loaded_normalization_dict = pickle.load(f)

        # Initialize the normalization layer
        input_normalizers_setting = dict()
        for input_name, input_type in self.event_info.input_types.items():
            input_normalizers_setting_local = {
                "log_mask": torch.tensor(
                    [feature_info.log_scale for feature_info in self.event_info.input_features[input_name]]),
                "mean": loaded_normalization_dict["input_mean"][input_name],
                "std": loaded_normalization_dict["input_std"][input_name]
            }

            if input_type in input_normalizers_setting:
                for element in input_normalizers_setting[input_type]:
                    input_normalizers_setting[input_type][element] = torch.cat(
                        input_normalizers_setting[input_type][element],
                        input_normalizers_setting_local[element],
                    )
            else:
                input_normalizers_setting[input_type] = input_normalizers_setting_local
        self.sequential_normalizer = normalizer(
            log_mask=input_normalizers_setting["SEQUENTIAL"]["log_mask"],
            mean=input_normalizers_setting["SEQUENTIAL"]["mean"],
            std=input_normalizers_setting["SEQUENTIAL"]["std"]
        )

        self.global_normalizer = normalizer(
            log_mask=input_normalizers_setting["GLOBAL"]["log_mask"],
            mean=input_normalizers_setting["GLOBAL"]["mean"],
            std=input_normalizers_setting["GLOBAL"]["std"]
        )

        self.global_input_dim = input_normalizers_setting["GLOBAL"]["log_mask"].size()[-1]
        self.sequential_input_dim = input_normalizers_setting["SEQUENTIAL"]["log_mask"].size()[-1]

        # Initialize the embedding layers

        self.global_embedding = GlobalVectorEmbedding(linear_block_type=self.options.Network.linear_block_type,
                                                      input_dim=self.global_input_dim,
                                                      hidden_dim_scale=self.options.Network.transformer_dim_scale,
                                                      initial_embedding_dim=self.options.Network.initial_embedding_dim,
                                                      final_embedding_dim=self.options.Network.hidden_dim,
                                                      normalization_type=self.options.Network.normalization,
                                                      activation_type=self.options.Network.linear_activation,
                                                      skip_connection=False,
                                                      num_embedding_layers=self.options.Network.num_embedding_layers,
                                                      dropout=self.options.Network.dropout)

        self.local_feature_indices = self.options.Network.local_point_index
        self.PET_body = PETBody(num_feat=len(self.local_feature_indices),
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
                                mode="train")
        self.combined_embedding = CombinedEmbedding(
            hidden_dim=self.options.Network.hidden_dim,
            position_embedding_dim=self.options.Network.position_embedding_dim,
        )

        self.object_encoder = ObjectEncoder(
            hidden_dim=self.options.Network.hidden_dim,
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
        )

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
        input_point_cloud = x['x']
        input_point_cloud_mask = x['x_mask'].unsqueeze(-1)
        global_conditions = x['conditions'].unsqueeze(1)  # (batch_size, 1, num_conditions)
        global_conditions_mask = x['conditions_mask'].unsqueeze(-1)  # (batch_size, 1)

        # Normalize the input point cloud
        input_point_cloud = self.sequential_normalizer(input_point_cloud, input_point_cloud_mask)
        global_conditions = self.global_normalizer(global_conditions, global_conditions_mask)

        # Embedding
        global_conditions = self.global_embedding(global_conditions, global_conditions_mask)
        local_feature = input_point_cloud[..., self.local_feature_indices]
        input_point_cloud = self.PET_body(local_feature, input_point_cloud, input_point_cloud_mask, time)

        # Object encoding for classification/assignment/regression
        embeddings, embeddings_mask = self.combined_embedding(
            x=input_point_cloud,
            y=global_conditions,
            x_mask=input_point_cloud_mask,
            y_mask=global_conditions_mask
        )
        embeddings_padding_mask = ~(embeddings_mask.squeeze(2).bool())
        embeddings, event_token = self.object_encoder(embeddings, embeddings_mask, embeddings_padding_mask)
        # Classification head
        classifications = self.class_head(event_token)

        return {
            "classifications": classifications,
        }

    def training_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = batch['x'].shape[0]
        time = batch['x'].new_ones((batch_size,))
        output = self.forward(batch, time)
        return output
