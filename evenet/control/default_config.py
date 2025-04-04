# --- Default Configuration --- #
_DEFAULTS = {
    "options": {

        # =========================================================================================
        # General Options
        # =========================================================================================

        # Maximum number of jets to consider per event
        "nMaxJet": 10,

        # =========================================================================================
        # Network Architecture
        # =========================================================================================

        "Network": {
            # Dimensions used internally by all hidden layers / transformers.
            "hidden_dim": 128,

            # DEPRECATED
            # Internal dimensions used during transformer and some linear layers.
            "transformer_dim": 128,

            # Internal dimensions used during transformer and some linear layers.
            # Scalar variant, multiply the input dim by this amount.
            "transformer_dim_scale": 2.0,

            # Hidden dimensionality of the first embedding layer.
            "initial_embedding_dim": 16,

            "position_embedding_dim": 32,

            # Maximum Number of double-sized embedding layers to add between the features and the encoder.
            # The size of the embedding dimension will be capped at the hidden_dim,
            # So setting this option to a very large integer will just keep embedding up to the hidden_dim.
            "num_embedding_layers": 10,

            # Number of encoder layers for the central shared transformer.
            "num_encoder_layers": 4,

            # Number of feed forward layers to add to branch heads.
            # Set to 0 to disable branch embedding layers.
            "num_branch_embedding_layers": 4,

            # Number of encoder layers for each of the quark branch transformers.
            # Set to 0 to disable branch encoder layers.
            "num_branch_encoder_layers": 4,

            # Number of extra linear layers before the attention layer when using split attention.
            # Only used if split_symmetric_attention is True
            # Set to 0 to disable jet embedding layers.
            "num_jet_embedding_layers": 0,

            # Number of extra transformer layers before the attention layer when using split attention.
            # Only used if split_symmetric_attention is True
            # Set to 0 to disable jet encoder layers.
            "num_jet_encoder_layers": 0,

            # Number of hidden layers to use for the particle classification head.
            "num_detection_layers": 1,

            # Number of hidden layers to use for the particle classification head.
            "num_regression_layers": 1,

            # Number of hidden layers to use for the particle classification head.
            "num_classification_layers": 1,

            # Whether to use a split approximate tensor attention layer.
            "split_symmetric_attention": True,

            # Number of heads for multi-head attention, used in all transformer layers.
            "num_attention_heads": 4,

            # Activation function for all transformer layers, 'relu' or 'gelu'.
            "transformer_activation": "gelu",

            # Whether to add skip connections to internal linear layers.
            # All layers support skip connections, this can turn them off.
            "skip_connections": True,

            # Whether to add skip connections to the initial set of embedding layers.
            "initial_embedding_skip_connections": True,

            # Structure for linear layers in the network
            #
            # Options are:
            # -------------------------------------------------
            # Basic
            # Resnet
            # Gated
            # GRU
            # -------------------------------------------------
            "linear_block_type": "GRU",

            # Structure for transformer layer
            #
            # Options are:
            # -------------------------------------------------
            # Standard
            # NormFirst
            # Gated
            # -------------------------------------------------
            "transformer_type": "Gated",

            # Non-linearity to use inside of the linear blocks.
            #
            # Options are:
            # -------------------------------------------------
            # None
            # ReLU
            # PReLU
            # ELU
            # GELU
            # -------------------------------------------------
            "linear_activation": "GELU",

            # Time Fourier Projection dimension
            "time_fprojection_dim": 64,

            # Local Embed to provide edge information for point cloud
            "enable_local_embedding": True,
            "num_local_layer": 3,
            "local_point_index": [2, 3],
            "local_Krank": 2,

            # Dropout added to all layers.
            "dropout": 0.0,

            # PET setting
            "PET_num_heads": 4,
            "PET_layer_scale": True,
            "PET_num_layers": 8,
            "PET_drop_probability": 0.0,
            "PET_dropout": 0.0,
            "PET_layer_scale_init": 1e-5,
            "PET_talking_head": False,

            # Resonance particle embed setting
            "feature_drop": 0.2,
            "num_feature_keep": 0,

            # Global variable generation setting
            "diff_sub_resnet_nlayer": 2,
            "diff_resnet_nlayer": 3,

            # Sequential variable generation setting
            "diff_transformer_nlayer": 2,

            # Whether to apply a normalization layer during linear / embedding layers.
            #
            # Options are:
            # -------------------------------------------------
            # None
            # BatchNorm
            # LayerNorm
            # MaskedBatchNorm
            # -------------------------------------------------
            "normalization": "LayerNorm",

            # What type of masking to use throughout the linear layers.
            #
            # Options are:
            # -------------------------------------------------
            # None
            # Multiplicative
            # Filling
            # -------------------------------------------------
            "masking": "Filling",

            # DEPRECATED
            # Whether to use PreLU activation on linear / embedding layers,
            # Otherwise a regular relu will be used.
            "linear_prelu_activation": True,
        },
        # =========================================================================================
        # Dataset Options
        # =========================================================================================

        "Dataset": {
            # Location of event ini file and the jet hdf5 files.
            # This is set by the constructor and should not be set manually.
            "resonance_info_file": None,
            "event_info_file": None,
            "training_file": None,
            "validation_file": None,
            "testing_file": None,

            # Whether to compute training data statistics to normalize features.
            "normalize_features": True,

            # Limit the dataset to this exact number of jets. Set to 0 to disable.
            "limit_to_num_jets": 0,

            # Whether to add weight to classes based on their training data prevalence.
            "balance_particles": False,

            # Whether to add a weight to the jet multiplicity to not forget about large events.
            "balance_jets": False,

            # Whether to add a weight to classification heads based on target presence.
            "balance_classifications": False,

            # Whether to train on partial events in the dataset.
            "partial_events": False,

            # Limit the dataset to the first x% of the data.
            "dataset_limit": 1.0,

            # Set a non-zero value here to deterministically shuffle the training dataset when selecting the subset.
            "dataset_randomization": 0,

            # Percent of data to use for training vs. validation.
            "train_validation_split": 0.95,

            # Percent of data to use for calculating statistics.
            "portion_for_statistics": 1.0,

            # Percent of data to use for calculating balance.
            "portion_for_balance": 1.0,

            # Normalization info file
            "normalization_file": None,
            "normalization_file_save_path": None,

            # Balance info file
            "balance_file": None,
            "balance_file_save_path": None,

            # Number of processes to spawn for data collection.
            "num_dataloader_workers": 4,
        },

        # =========================================================================================
        # Training Options
        # =========================================================================================

        "Training": {
            # Whether to mask vectors not in the events during operation.
            # Should most-definitely be True, but this is here for testing.
            "mask_sequence_vectors": True,
            # Training batch size.
            "batch_size": 4096,

            # Whether we should combine the two possible targets: swapped and not-swapped.
            # If None, then we will only use the proper target ordering.
            #
            # Options are:
            # -------------------------------------------------
            # None
            # min
            # softmin
            # mean
            # -------------------------------------------------
            "combine_pair_loss": "min",

            # The optimizer to use for training the network.
            # This must be a valid class in torch.optim or nvidia apex with 'apex' prefix.
            "optimizer": "AdamW",

            # Optimizer learning rate.
            "learning_rate": 0.001,

            # Optimizer setting for fine tune
            "learning_rate_factor": 1.0,
            "fine_tune": False,

            # Gamma exponent for focal loss. Setting it to 0.0 will disable focal loss and use regular cross-entropy.
            "focal_gamma": 0.0,

            # Combinatorial offset for the masked softmax discrepancy
            "combinatorial_scale": 0.0,

            # Number of epochs to ramp up the learning rate up to the given value. Can be fractional.
            "learning_rate_warm_up_factor": 3.0,

            # Number of times to cycle the learning rate through cosine annealing with hard resets.
            # Set to 0 to disable cosine annealing and just use a decaying learning rate.
            "learning_rate_cycles": 0,

            # Scalar term for the primary jet assignment loss.
            "assignment_loss_scale": 1.0,

            # Scalar term for the direct classification loss of particles.
            "detection_loss_scale": 0.0,

            # Scalar term for the symmetric KL-divergence loss between distributions.
            "kl_loss_scale": 0.0,

            # Scalar term for regression L2 loss term
            "regression_loss_scale": 0.0,

            # Scalar term for classification Cross Entropy loss term
            "classification_loss_scale": 0.0,

            # Scalar term for generation
            "generation_loss_scale": 0.0,  # TODO: turn to zero after testing
            "feature_generation_loss_scale": 0.0,  # TODO: turn to zero after testing

            # Automatically balance loss terms using Jacobians.
            "balance_losses": True,

            # Optimizer l2 penalty based on weight values.
            "l2_penalty": 0.0,

            # Clip the L2 norm of the gradient. Set to 0.0 to disable.
            "gradient_clip": 0.0,


            # Number of epochs to train for.
            "epochs": 100,

            # Total number of GPUs to use.
            "num_gpu": 1,

            # Total number of nodes to use.
            "num_node": 1,
        },

        # =========================================================================================
        # Miscellaneous Options
        # =========================================================================================

        "Misc": {
            # Whether to print additional information during training and log extra metrics.
            "verbose_output": True,

            # Misc parameters used by sherpa to delegate GPUs and output directories.
            # These should not be set manually.
            "usable_gpus": "",
            "trial_time": "",
            "trial_output_dir": "./test_output"
        }
    },
}
