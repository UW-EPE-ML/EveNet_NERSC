import pytorch_lightning as pl
import numpy as np
import ray
import torch
from torch import nn
from collections import OrderedDict

# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from evenet.control.config import DotDict
from evenet.network.learning_rate_schedules import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from torch.profiler import profile, ProfilerActivity
import os, pickle

class JetReconstructionBase(pl.LightningModule):
    def __init__(self, config: DotDict, total_events: int = None):
        super(JetReconstructionBase, self).__init__()

        # Basic configuration
        self.options = config.options
        self.save_hyperparameters(self.options)
        self.event_info = config.event_info

        # Helper arrays for permutation groups. Used for the partial-event loss functions.
        self.event_permutation_tensor = OrderedDict()
        for process in self.event_info.event_permutation_group:
            event_permutation_group = np.array(self.event_info.event_permutation_group[process])
            self.event_permutation_tensor[process] = torch.nn.Parameter(torch.from_numpy(event_permutation_group),
                                                                        False)


        # Datasets
        # self.training_dataset, self.validation_dataset, self.testing_dataset = self.create_datasets()

        # Compute class weights for particles from the training dataset target distribution
        self.balance_particles = False
        self.particle_index_tensor = OrderedDict()
        self.particle_weights_tensor = OrderedDict()


        # Load the balance file if it exists
        balance_dict = dict()
        if self.options.Dataset.balance_file is not None:
            with open(self.options.Dataset.balance_file, "rb") as f:
                loaded_balance_dict = pickle.load(f)

        if self.options.Dataset.balance_particles and self.options.Dataset.partial_events:
            print("calculating balance for particle")
            if self.options.Dataset.balance_file is None:
                # balance_dict["particle_balance"] = self.training_dataset.compute_particle_balance(portion = self.options.Dataset.portion_for_balance)
                pass
            else:
                balance_dict["particle_balance"] = loaded_balance_dict["particle_balance"]

            for process in balance_dict["particle_balance"]:
                index_tensor, weights_tensor = balance_dict["particle_balance"][process]
                self.particle_index_tensor[process] = torch.nn.Parameter(index_tensor, requires_grad=False)
                self.particle_weights_tensor[process] = torch.nn.Parameter(weights_tensor, requires_grad=False)
            self.balance_particles = True

        self.particle_index_tensor = nn.ParameterDict(self.particle_index_tensor)
        self.particle_weights_tensor = nn.ParameterDict(self.particle_weights_tensor)
        self.event_permutation_tensor = nn.ParameterDict(self.event_permutation_tensor)

        # Compute class weights for jets from the training dataset target distribution
        self.balance_jets = False
        if self.options.Dataset.balance_jets:
            print("calculating balance for num jets")
            if self.options.Dataset.balance_file is None:
                # jet_weights_tensor = self.training_dataset.compute_vector_balance(portion = self.options.Dataset.portion_for_balance)
                pass
            else:
                jet_weights_tensor = loaded_balance_dict["jet_balance"]

            balance_dict["jet_balance"] = jet_weights_tensor
            self.jet_weights_tensor = torch.nn.Parameter(jet_weights_tensor, requires_grad=False)
            self.balance_jets = True

        self.balance_classifications = self.options.Dataset.balance_classifications
        if self.balance_classifications:
            print("calculating balance for classifications")
            if self.options.Dataset.balance_file is None:
                # classification_weights = {
                #     key: torch.nn.Parameter(value, requires_grad=False)
                #     for key, value in self.training_dataset.compute_classification_balance(portion = self.options.Dataset.portion_for_balance).items()
                # }
                pass
            else:
                classification_weights = loaded_balance_dict["class_balance"]

            balance_dict["class_balance"] = classification_weights

            self.classification_weights = torch.nn.ParameterDict(classification_weights)

        # Helper variables for keeping track of the number of batches in each epoch.
        # Used for learning rate scheduling and other things.
        self.steps_per_epoch = total_events // (self.options.Training.batch_size * max(1, self.options.Training.num_gpu) * max(1, self.options.Training.num_node))
        self.total_steps = self.steps_per_epoch * self.options.Training.epochs
        self.warmup_steps = int(round(self.steps_per_epoch * self.options.Training.learning_rate_warmup_epochs))

        if self.options.Dataset.balance_file_save_path is not None:
            dir_path  = '/'.join(self.options.Dataset.balance_file_save_path.split('/')[:-1])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(self.options.Dataset.balance_file_save_path, "wb") as f:
                pickle.dump(balance_dict, f)

    @property
    def dataset(self):
        return ray.data

    @property
    def dataloader(self):
        return DataLoader

    @property
    def dataloader_options(self):
        return {
            "batch_size": self.options.Training.batch_size,
            "pin_memory": self.options.Training.num_gpu > 0,
            "num_workers": self.options.Training.num_dataloader_workers,
            "persistent_workers": True
        }

    # def create_datasets(self):
    #     event_info_file = self.options.Dataset.event_info_file
    #     resonance_info_file = self.options.Dataset.resonance_info_file
    #     training_file = self.options.Dataset.training_file
    #     validation_file = self.options.Dataset.validation_file
    #
    #     training_range = self.options.Dataset.dataset_limit
    #     validation_range = 1.0
    #
    #     # If we dont have a validation file provided, create one from the training file.
    #     if len(validation_file) == 0:
    #         validation_file = training_file
    #
    #         # Compute the training / validation ranges based on the data-split and the limiting percentage.
    #         train_validation_split = self.options.Dataset.dataset_limit * self.options.Dataset.train_validation_split
    #         training_range = (0.0, train_validation_split)
    #         validation_range = (train_validation_split, self.options.Dataset.dataset_limit)
    #
    #     # Construct primary training datasets
    #     # Note that only the training dataset should be limited to full events or partial events.
    #     training_dataset = self.dataset(
    #         data_file=training_file,
    #         event_info=event_info_file,
    #         limit_index=training_range,
    #         vector_limit=self.options.Dataset.limit_to_num_jets,
    #         partial_events=self.options.Dataset.partial_events,
    #         randomization_seed=self.options.Dataset.dataset_randomization,
    #         resonance_info=resonance_info_file
    #     )
    #
    #     validation_dataset = self.dataset(
    #         data_file=validation_file,
    #         event_info=event_info_file,
    #         limit_index=validation_range,
    #         vector_limit=self.options.Dataset.limit_to_num_jets,
    #         randomization_seed=self.options.Dataset.dataset_randomization,
    #         resonance_info=resonance_info_file
    #     )
    #
    #     # Optionally construct the testing dataset.
    #     # This is not used in the main training script but is still useful for testing later.
    #     testing_dataset = None
    #     if len(self.options.Dataset.testing_file) > 0:
    #         testing_dataset = self.dataset(
    #             data_file=self.options.Dataset.testing_file,
    #             event_info=self.options.Dataset.event_info_file,
    #             limit_index=self.options.Dataset.dataset_limit, # TODO: Allow modified testing range
    #             vector_limit=self.options.Dataset.limit_to_num_jets,
    #             resonance_info=resonance_info_file
    #         )
    #
    #     return training_dataset, validation_dataset, testing_dataset

    def configure_optimizers(self):
        optimizer = None

        if 'apex' in self.options.Training.optimizer:
            try:
                # noinspection PyUnresolvedReferences
                import apex.optimizers

                if self.options.Training.optimizer == 'apex_adam':
                    optimizer = apex.optimizers.FusedAdam

                elif self.options.Training.optimizer == 'apex_lamb':
                    optimizer = apex.optimizers.FusedLAMB

                else:
                    optimizer = apex.optimizers.FusedSGD

            except ImportError:
                pass

        else:
            optimizer = getattr(torch.optim, self.options.Training.optimizer)

        if optimizer is None:
            print(f"Unable to load desired optimizer: {self.options.Training.optimizer}.")
            print(f"Using pytorch AdamW as a default.")
            optimizer = torch.optim.AdamW

        decay_mask = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [param for name, param in self.named_parameters()
                           if not any(no_decay in name for no_decay in decay_mask) and ('vector_embedding' in name)],
                "weight_decay": self.options.Training.l2_penalty,
                "lr": self.options.Training.learning_rate * self.options.Training.learning_rate_factor,
            },
            {
                "params": [param for name, param in self.named_parameters()
                           if not any(no_decay in name for no_decay in decay_mask) and not ('vector_embedding' in name)],
                "weight_decay": self.options.Training.l2_penalty,
                "lr": self.options.Training.learning_rate,
            },
            {
                "params": [param for name, param in self.named_parameters()
                           if any(no_decay in name for no_decay in decay_mask) and ('vector_embedding' in name)],
                "weight_decay": 0.0,
                "lr": self.options.Training.learning_rate * self.options.Training.learning_rate_factor,
            },
            {
                "params": [param for name, param in self.named_parameters()
                           if any(no_decay in name for no_decay in decay_mask) and not ('vector_embedding' in name)],
                "weight_decay": 0.0,
                "lr": self.options.Training.learning_rate
            }
        ]
        optimizer = optimizer(optimizer_grouped_parameters, lr=self.options.Training.learning_rate)


        if self.options.Training.learning_rate_cycles < 1:
            scheduler = get_linear_schedule_with_warmup(
                 optimizer,
                 num_warmup_steps=self.warmup_steps,
                 num_training_steps=self.total_steps
             )
        else:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.options.Training.learning_rate_cycles
            )

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.training_dataset, shuffle=False, drop_last=True, **self.dataloader_options) # TODO: testing with shuffle

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.validation_dataset, drop_last=True, **self.dataloader_options)

    def test_dataloader(self) -> DataLoader:
        if self.testing_dataset is None:
            raise ValueError("Testing dataset not provided.")

        return self.dataloader(self.testing_dataset, **self.dataloader_options)
