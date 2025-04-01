import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import re
import os, pickle

from evenet.control.config import Config, DotDict
from evenet.dataset.types import Tuple, Outputs, Source, Predictions, DistributionInfo, InputType, Batch, \
    AssignmentTargets
from evenet.dataset.regressions import regression_loss
from typing import Dict, List

from evenet.network.layers.linear_block.regularization import RandomDrop
from evenet.network.layers.vector_encoder import JetEncoder
from evenet.network.layers.branch_decoder import BranchDecoder
from evenet.network.layers.embedding import MultiInputVectorEmbedding
from evenet.network.layers.embedding.local_embedding import LocalEmbedding
from evenet.network.layers.regression_decoder import RegressionDecoder
from evenet.network.layers.classification_decoder import ClassificationDecoder
from evenet.network.layers.event_generation_decoder import EventGenerationDecoder
from evenet.network.layers.jet_generation_decoder import JetGenerationDecoder
from evenet.network.layers.feature_generation_decoder import FeatureGenerationDecoder
from evenet.network.layers.embedding.normalizer import Normalizer

from evenet.network.prediction_selection import extract_predictions
from evenet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase
from evenet.network.layers.diffusion.sampler import Diffusion_Sampler
from collections import OrderedDict

TArray = np.ndarray

from evenet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence


def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list

    return output


class JetReconstructionNetwork(JetReconstructionBase):
    def __init__(self, config: DotDict, torch_script: bool = False, **kwargs):
        """ Base class defining the SPANet architecture.

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(JetReconstructionNetwork, self).__init__(config, **kwargs)

        compile_module = torch.jit.script if torch_script else lambda x: x

        self.hidden_dim = self.options.Network.hidden_dim

        # Normalizer
        print("start to init network (calculating input statistics)")

        if self.options.Dataset.normalization_file is not None:
            with open(self.options.Dataset.normalization_file, "rb") as f:
                normalization_dict = pickle.load(f)

        self.num_vector_normalizer = []
        self.normalizer = []

        for name, source in normalization_dict["input_num_mean"].items():
            self.num_vector_normalizer.append(Normalizer(source, normalization_dict["input_num_std"][name]))
        self.num_vector_normalizer = nn.ModuleList(self.num_vector_normalizer)

        for name, source in normalization_dict["input_mean"].items():
            self.normalizer.append(Normalizer(source, normalization_dict["input_std"][name]))

        self.normalizer = nn.ModuleList(self.normalizer)

        print("start to init network (calculating regressions statistics)")

        if self.options.Dataset.normalization_file_save_path is not None:
            dir_path = '/'.join(self.options.Dataset.normalization_file_save_path.split('/')[:-1])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(self.options.Dataset.normalization_file_save_path, "wb") as f:
                pickle.dump(normalization_dict, f)
        feature_means = torch.cat([mean_.unsqueeze(0) for mean_ in normalization_dict["regression_mean"].values()],
                                  dim=-1) if len(normalization_dict["regression_mean"].values()) > 1 else None
        feature_stds = torch.cat([std_.unsqueeze(0) for std_ in normalization_dict["regression_std"].values()],
                                 dim=-1) if len(normalization_dict["regression_std"].values()) > 1 else None
        self.feature_normalizer = Normalizer(feature_means,
                                             feature_stds) if feature_means is not None else nn.Identity()

        print("finish statistics calculation")

        print("configure resonance_particle_preperties_normalizer")
        self.resonance_particle_properties_normalizer = Normalizer(self.event_info.resonance_particle_properties_mean,
                                                                   self.event_info.resonance_particle_properties_std)
        self.num_resonance_particle_feature = self.event_info.resonance_particle_properties_mean.size(0)

        print("configure resonance_particle_properties param")
        self.resonance_particle_properties = nn.ParameterDict({topology_name:
            nn.Parameter(
                self.event_info.pairing_topology[topology_name][
                    "resonance_particle_properties"],
                requires_grad=False)
            for topology_name in self.event_info.pairing_topology})

        print("configure input embedding")
        self.embedding = compile_module(MultiInputVectorEmbedding(
            self.options,
            self.event_info,
            mean=normalization_dict["input_mean"],
            std=normalization_dict["input_std"]
        ))

        print("configure input encoder")
        self.encoder = compile_module(JetEncoder(
            self.options,
        ))

        print("configure condition_random_drop")
        self.condition_random_drop = RandomDrop(
            self.options.Network.feature_drop,
            self.options.Network.num_feature_keep
        )
        self.condition_embed = nn.Sequential(
            nn.Linear(self.num_resonance_particle_feature, self.options.Network.hidden_dim),
            nn.GELU(),
            nn.Linear(self.options.Network.hidden_dim, self.options.Network.hidden_dim)
        )

        print("configure condition_branch decoder")
        self.branch_decoders = nn.ModuleDict({topology_name:
            BranchDecoder(
                self.options,
                self.event_info.pairing_topology_category[topology_name]["product_particles"].names,
                self.event_info.pairing_topology_category[topology_name]["product_symmetry"],
                self.enable_softmax
            )
            for topology_name in self.event_info.pairing_topology_category})

        self.permutation_indices = dict()
        self.num_targets = dict()

        print("configure permutation indices")
        for process in self.event_info.process_names:
            self.permutation_indices[process] = []
            self.num_targets[process] = []
            for event_particle_name, product_symmetry in self.event_info.product_symmetries[process].items():
                topology_name = ''.join(self.event_info.product_particles[process][event_particle_name].names)
                topology_name = f"{event_particle_name}/{topology_name}"
                topology_name = re.sub(r'\d+', '', topology_name)
                topology_category_name = self.event_info.pairing_topology[topology_name]["pairing_topology_category"]
                self.permutation_indices[process].append(
                    self.branch_decoders[topology_category_name].permutation_indices)
                self.num_targets[process].append(self.branch_decoders[topology_category_name].num_targets)

        print("configure regression decoder")
        self.regression_decoder = compile_module(RegressionDecoder(
            self.options,
            self.event_info,
            means=normalization_dict["regression_mean"],
            stds=normalization_dict["regression_std"]
        ))

        print("configure classification_decoder")
        self.classification_decoder = compile_module(ClassificationDecoder(
            self.options,
            self.event_info
        ))

        print("configure event_generation_decoder")
        self.event_generation_decoder = compile_module(EventGenerationDecoder(
            self.options,
            self.event_info
        ))

        print("configure diffusion sampler")
        self.diffusion_sampler = Diffusion_Sampler(self.options,
                                                   self.event_info,
                                                   normalization_dict["input_num_mean"])

        print("configure jet generation_decoder")
        self.jet_generation_decoder = compile_module(JetGenerationDecoder(
            self.options,
            self.diffusion_sampler.num_sequential,
            self.diffusion_sampler.num_global
        ))

        print("configure feature generation decoder")
        self.feature_generation_decoder = compile_module(FeatureGenerationDecoder(
            self.options,
            cond_dim=self.event_info.num_regressions,
            out_dim=self.event_info.num_regressions,
        ))

        self.input_features = self.event_info.input_features
        self.input_types = self.diffusion_sampler.input_types
        self.output_dim_mapping = self.diffusion_sampler.output_dim_mapping
        self.process_names = self.event_info.process_names

        print("finish init model")

        # An example input for generating the network's graph, batch size of 2
        # self.example_input_array = tuple(x.contiguous() for x in self.training_dataset[:2][0])

    @property
    def enable_softmax(self):
        return True

    def forward(self, sources: Tuple[Source, ...], source_time: Tensor, source_num_vector: Dict[str, Tensor],
                regressions: Dict[str, Source] = None, mode: str = "classifier") -> Outputs:

        ############################################################
        ## Perform normalization & Add perturbation for diffusion ##
        ############################################################

        sources = self.diffusion_sampler.normalize(sources, self.normalizer)
        batch_size = source_time.size(0)

        if (mode == 'classifier'):
            sources_seq_perturbed = sources

        else:
            # Perturbed only global source
            sources_global_perturbed, sources_score_global_perturbed = self.diffusion_sampler.add_perturbation(sources,
                                                                                                               source_time,
                                                                                                               InputType.Global)  # Output results are already normalized
            # Perturbed only sequential, source
            sources_seq_perturbed, sources_score_seq_perturbed = self.diffusion_sampler.add_perturbation(sources,
                                                                                                         source_time,
                                                                                                         InputType.Sequential)  # Output are already normalized
            # Perturb number of vectors(jets)
            perturbed_source_num_vector, source_num_vector_score = self.diffusion_sampler.add_perturbation_dict(
                source_num_vector, source_time, self.num_vector_normalizer)  # Output are already normalized

            # Perturb regression features
            if (self.options.Training.feature_generation_loss_scale > 0):
                perturbed_feature, feature_score = self.diffusion_sampler.add_perturbation_source_dict(regressions,
                                                                                                       source_time,
                                                                                                       self.feature_normalizer)

            # Combine global information for the input of global generation head
            x_global_perturbed, score_global_perturbed = self.diffusion_sampler.prepare_format(sources_global_perturbed,
                                                                                               source_time,
                                                                                               perturbed_source_num_vector,
                                                                                               sources_score_global_perturbed,
                                                                                               source_num_vector_score)
            x_seq_perturbed, score_seq_perturbed = self.diffusion_sampler.prepare_format(sources_seq_perturbed,
                                                                                         source_time,
                                                                                         perturbed_source_num_vector,
                                                                                         sources_score_seq_perturbed,
                                                                                         source_num_vector_score)
        # Embed all of the different input regression_vectors into the same latent space.
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources_seq_perturbed, source_time)

        # Extract features from data using transformer
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)

        if (mode == 'classifier'):
            pred_score = {"Global": None, "Sequential": None, "Feature": None}
            true_score = {"Global": None, "Sequential": None, "Feature": None}
        else:
            pred_v_global = self.event_generation_decoder(x_global_perturbed["Global"],
                                                          source_time)  # pred_v_global: Source(data, mask)
            pred_v_sequential = self.jet_generation_decoder(embeddings, source_time, padding_masks, sequence_masks,
                                                            global_masks)
            embeddings_unperturbed, _, _, _ = self.embedding(sources, torch.zeros_like(source_time))

            pred_score = dict()
            true_score = dict()
            if (self.options.Training.feature_generation_loss_scale > 0):
                pred_v_feature = Source(
                    self.feature_generation_decoder(embeddings_unperturbed, perturbed_feature.data, source_time,
                                                    padding_masks, sequence_masks), perturbed_feature.mask)
                pred_score["Feature"] = pred_v_feature
                true_score["Feature"] = feature_score
            else:
                pred_score["Feature"] = None
                true_score["Feature"] = None

            pred_score["Global"] = pred_v_global
            pred_score["Sequential"] = pred_v_sequential

            true_score["Global"] = score_global_perturbed["Global"]
            true_score["Sequential"] = score_seq_perturbed["Sequential"]

        # Create output lists for each particle in event.
        assignments = dict()
        detections = dict()

        encoded_vectors = {
            "EVENT": event_vector
        }

        # Pass the shared hidden state to every decoder branch

        branch_decoder_result = dict()
        for topology_name in self.event_info.pairing_topology:
            topology_category_name = self.event_info.pairing_topology[topology_name]["pairing_topology_category"]
            condition_variable = self.resonance_particle_properties_normalizer(
                self.resonance_particle_properties[topology_name])
            condition_variable = condition_variable.expand(1, batch_size, condition_variable.size(0))
            condition_variable = self.condition_embed(self.condition_random_drop(condition_variable))

            decoder = self.branch_decoders[topology_category_name]
            (
                assignment,
                detection,
                assignment_mask,
                event_particle_vector,
                product_particle_vectors
            ) = decoder(hidden, padding_masks, sequence_masks, global_masks, condition_variable)
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

                # Assign the summarising vectors to their correct structure.
                # encoded_vectors["/".join([decoder.particle_name, "PARTICLE"])] = event_particle_vector
                # for product_name, product_vector in zip(decoder.product_names, product_particle_vectors):
                #    encoded_vectors["/".join([decoder.particle_name, product_name])] = product_vector

        # Predict the valid regressions for any real values associated with the event.
        regressions = self.regression_decoder(encoded_vectors)
        # Predict additional classification targets for any branch of the event.
        classifications = self.classification_decoder(encoded_vectors)

        return Outputs(
            assignments,
            detections,
            encoded_vectors,
            regressions,
            classifications,
            true_score,
            pred_score
        )

    def predict(self, sources: Tuple[Source, ...], source_time: Tensor,
                source_num_vectors: Dict[str, Tensor]) -> Predictions:
        with torch.no_grad():
            outputs = self.forward(sources, source_time, source_num_vectors)

            # Extract assignment probabilities and find the least conflicting assignment.
            assignments = {process: extract_predictions([
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in outputs.assignments[process]
            ])
                for process in outputs.assignments
            }

            # Convert detection logits into probabilities and move to CPU.
            detections = {process: np.stack([
                torch.sigmoid(detection).cpu().numpy()
                for detection in outputs.detections[process]
            ])
                for process in outputs.detections
            }

            # Move regressions to CPU and away from torch.
            regressions = {
                key: value.cpu().numpy()
                for key, value in outputs.regressions.items()
            }

            classifications = {
                key: value.cpu().argmax(1).numpy()
                for key, value in outputs.classifications.items()
            }

        return Predictions(
            assignments,
            detections,
            regressions,
            classifications
        )

    def predict_assignments(self, sources: Tuple[Source, ...]) -> np.ndarray:
        # Run the base prediction step
        with torch.no_grad():
            assignments = [
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in self.forward(sources).assignments
            ]

        # Find the optimal selection of jets from the output distributions.
        return extract_predictions(assignments)

    def predict_assignments_and_detections(self, sources: Tuple[Source, ...]) -> Tuple[TArray, TArray]:
        assignments, detections, regressions, classifications = self.predict(sources)

        # Always predict the particle exists if we didn't train on it
        if self.options.Training.detection_loss_scale == 0:
            detections += 1

        return assignments, detections >= 0.5

    def predict_sequential_vector(self, sources: Tuple[Source, ...], source_time: Tensor,
                                  source_num_vector: Dict[str, Tensor]) -> Tuple[Source, ...]:

        # sources = self.diffusion_sampler.normalize(sources, self.normalizer)
        x_seq_perturbed, _ = self.diffusion_sampler.prepare_format(sources, source_time, source_num_vector, sources,
                                                                   source_num_vector)
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources, source_time)
        pred_v_sequential = self.jet_generation_decoder(embeddings, source_time, padding_masks, sequence_masks,
                                                        global_masks)

        output_v = []

        for input_index, name in enumerate(self.input_types):
            data, mask = sources[input_index]
            if (self.input_types[name] == InputType.Global):
                output_v.append(Source(torch.zeros_like(data), mask))
            else:
                pred_v = pred_v_sequential[0][..., self.output_dim_mapping["Sequential"][name]]
                output_v.append(Source(pred_v, mask))

        return tuple(output_v), global_masks

    def predict_feature_vector(self, sources: Tuple[Source, ...], source_time: Tensor, regressions: Tensor) -> Tuple[
        Source, ...]:
        sources = self.diffusion_sampler.normalize(sources, self.normalizer)
        embeddings_unperturbed, padding_masks, sequence_masks, global_masks = self.embedding(sources, torch.zeros_like(
            source_time))
        pred_v_feature = self.feature_generation_decoder(embeddings_unperturbed, regressions, source_time,
                                                         padding_masks, sequence_masks)
        return pred_v_feature

    def generate_feature(self, sources, batch_size: int, num_steps=100, num_iteration=1):
        Generated_Feature = []
        for iteration in range(num_iteration):
            Generated_Feature.append(
                self.diffusion_sampler.generate_feature(self, sources, batch_size, self.event_info.num_regressions,
                                                        self.feature_normalizer, num_steps=num_steps, device='cuda'))

        Generated_Feature = torch.stack(Generated_Feature, dim=-1)  # [B, iter]
        Generated_output = OrderedDict()
        for regression_index, regression_target in enumerate(self.event_info.regressions):
            Generated_output[regression_target] = Generated_Feature[:, regression_index, :].cpu().numpy()
        return Generated_output

    def generate(self, batch_size: int, sources=None, sources_num_vector=None):
        Generated_Event = self.diffusion_sampler.generate(self, batch_size, self.normalizer, self.num_vector_normalizer,
                                                          num_steps=200, device='cuda', input_sources=sources,
                                                          sources_num_vector=sources_num_vector)
        return Generated_Event

    def get_reference_sample(self, sources: Tuple[Source, ...], source_time: Tensor,
                             source_num_vector: Dict[str, Tensor]) -> Outputs:

        sources = self.diffusion_sampler.normalize(sources, self.normalizer)
        perturbed_sources, sources_score = self.diffusion_sampler.add_perturbation(sources, source_time,
                                                                                   InputType.Global)  # Output results are already normalized
        perturbed_source_num_vector, source_num_vector_score = self.diffusion_sampler.add_perturbation_dict(
            source_num_vector, source_time, self.num_vector_normalizer,
            device=self.device)  # Output are already normalized
        perturbed_x, perturbed_score = self.diffusion_sampler.prepare_format(perturbed_sources, source_time,
                                                                             perturbed_source_num_vector, sources_score,
                                                                             source_num_vector_score)

        sample = self.diffusion_sampler.decode_tensor(perturbed_x["Global"], "Global")
        sample_seq = self.diffusion_sampler.decode_tensor(perturbed_x["Sequential"], "Sequential")
        sample = self.diffusion_sampler.denormalizer(sample, sample_seq, self.normalizer, self.num_vector_normalizer)
        return sample

    #########################################################################
    ##  Loss Func 
    #########################################################################

    def particle_symmetric_loss(self, assignment: Tensor, detection: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, self.options.Training.focal_gamma)
        detection_loss = F.binary_cross_entropy_with_logits(detection, mask.float(), reduction='none')

        return torch.stack((
            self.options.Training.assignment_loss_scale * assignment_loss,
            self.options.Training.detection_loss_scale * detection_loss
        ))

    def compute_symmetric_losses(self, assignments: List[Tensor], detections: List[Tensor], process: str, targets):

        symmetric_losses = []

        # TODO think of a way to avoid this memory transfer but keep permutation indices synced with checkpoint
        # Compute a separate loss term for every possible target permutation.
        for permutation in self.event_permutation_tensor[process].cpu().numpy():
            # Find the assignment loss for each particle in this permutation.
            current_permutation_loss = tuple(
                self.particle_symmetric_loss(assignment, detection, target, mask)
                for assignment, detection, (target, mask)
                in zip(assignments, detections, targets[permutation])
            )

            # The loss for a single permutation is the sum of particle losses.
            symmetric_losses.append(
                torch.stack(current_permutation_loss))  # Shape: (NUM_PERMUTATIONS, NUM_PARTICLES, 2, BATCH_SIZE)

        return torch.stack(symmetric_losses)

    def combine_symmetric_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        # Default option is to find the minimum loss term of the symmetric options.
        # We also store which permutation we used to achieve that minimal loss.
        # combined_loss, _ = symmetric_losses.min(0)
        total_symmetric_loss = symmetric_losses.sum((1, 2))
        index = total_symmetric_loss.argmin(0)

        combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]

        # Simple average of all losses as a baseline.
        if self.options.Training.combine_pair_loss.lower() == "mean":
            combined_loss = symmetric_losses.mean(0)

        # Soft minimum function to smoothly fuse all loss function weighted by their size.
        if self.options.Training.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(total_symmetric_loss, 0)
            weights = weights.unsqueeze(1).unsqueeze(1)
            combined_loss = (weights * symmetric_losses).sum(0)
        return combined_loss, index

    def symmetric_losses(
            self,
            assignments: List[Tensor],
            detections: List[Tensor],
            targets: Tuple[Tuple[Tensor, Tensor], ...],
            process: str
    ) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle

        symmetric_losses_summary = OrderedDict()
        assignments = [prediction + torch.log(torch.scalar_tensor(num_targets))
                       for prediction, num_targets in zip(assignments, self.num_targets[process])]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = numpy_tensor_array(targets)

        # Compute the loss on every valid permutation of the targets

        # Squash the permutation losses into a single value.
        symmetric_losses = self.compute_symmetric_losses(assignments, detections, process, targets)
        return self.combine_symmetric_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[Tensor], masks: Tensor, process: str) -> Tensor:
        divergence_loss = []

        for i, j in self.event_info.event_transpositions[process]:
            # Symmetric divergence between these two distributions
            div = jensen_shannon_divergence(predictions[process][i], predictions[process][j])

            # ERF term for loss
            loss = torch.exp(-(div ** 2))
            loss = loss.masked_fill(~masks[i], 0.0)
            loss = loss.masked_fill(~masks[j], 0.0)
            divergence_loss.append(loss)

        return torch.stack(divergence_loss).mean(0)

    def add_kl_loss(
            self,
            total_loss: List[Tensor],
            assignments: List[Tensor],
            masks: Tensor,
            weights: Tensor,
            mode: str,
            process: str,
            status: str
    ) -> List[Tensor]:

        if len(self.event_info.event_transpositions[process]) == 0:
            return total_loss

        # Compute the symmetric loss between all valid pairs of distributions.
        kl_loss = self.symmetric_divergence_loss(assignments, masks, process)
        kl_loss = (weights * kl_loss).sum() / masks.sum()

        with torch.no_grad():
            self.log(f"{status}/loss/{mode}/{process}/symmetric_loss", kl_loss, sync_dist=True)
            if torch.isnan(kl_loss):
                raise ValueError("Symmetric KL Loss has diverged.")

        return total_loss + [self.options.Training.kl_loss_scale * kl_loss]

    def add_regression_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets: Dict[str, Tensor],
            weights: Tensor,
            mode: str,
            status: str
    ) -> List[Tensor]:

        # weights: [B, 1]

        regression_terms = []

        for key in targets:
            current_target_type = self.event_info.regression_types[key]
            current_prediction = predictions[key]
            current_target = targets[key].data
            current_target_mask = targets[key].mask

            current_mean = self.regression_decoder.means[key]
            current_std = self.regression_decoder.stds[key]

            current_mask = (~torch.isnan(current_target) & current_target_mask)

            current_loss = regression_loss(current_target_type)(
                current_prediction[current_mask],
                current_target[current_mask],
                current_mean,
                current_std
            )
            current_loss = current_loss * weights[current_mask]
            if current_loss.size(0) > 0:
                current_loss = torch.nanmean(current_loss / float(len(targets)))
            else:
                current_loss = torch.tensor(0.0, device=current_loss.device)

            with torch.no_grad():
                self.log(f"{status}/loss/{mode}/regression/{key}", current_loss, sync_dist=True)

            regression_terms.append(self.options.Training.regression_loss_scale * current_loss)

        return total_loss + regression_terms

    def add_classification_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets: Dict[str, Tensor],
            weights: Tensor,
            mode: str,
            status: str
    ) -> List[Tensor]:
        classification_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            weight = None if not self.balance_classifications else self.classification_weights[key]
            current_loss = F.cross_entropy(
                current_prediction,
                current_target,
                ignore_index=-1,
                weight=weight,
                reduction='none'
            )
            current_loss = torch.mean(current_loss * weights)
            classification_terms.append(
                self.options.Training.classification_loss_scale * current_loss / float(len(targets)))

            with torch.no_grad():
                self.log(f"{status}/loss/{mode}/classification/{key}", current_loss, sync_dist=True)

        return total_loss + classification_terms

    def add_generation_loss(
            self,
            total_loss: List[Tensor],
            predict_score: Tuple[Tensor, Tensor],
            target_score: Tuple[Tensor, Tensor],
            name: str,
            status: str
    ) -> List[Tensor]:

        generation_terms = []

        v_prediction, mask_prediction = predict_score
        v_target, mask_target = target_score

        if name == "Sequential":
            current_loss = torch.sum(((v_prediction - v_target) ** 2) * (mask_prediction.float())) / (
                torch.sum((mask_prediction.float()))) / v_prediction.shape[-1]
        elif name == "Feature":
            current_mask = mask_prediction[..., 0] & mask_target[..., 0] & ~torch.isnan(v_target).any(
                dim=1) & ~torch.isnan(v_prediction).any(dim=1)
            v_prediction = v_prediction[current_mask]
            v_target = v_target[current_mask]
            current_loss = torch.mean((v_prediction - v_target) ** 2)
        else:
            current_loss = torch.mean((v_prediction - v_target) ** 2)
        generation_terms.append(self.options.Training.generation_loss_scale * current_loss)
        with torch.no_grad():
            self.log(f"{status}/loss/generation/{name}", current_loss, sync_dist=True)
        return total_loss + generation_terms

    def loss_func(self, batch: Batch, batch_nb: int, mode="classifier", status="train") -> Dict[str, Tensor]:
        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        # print(f"Rank {self.global_rank}: Start loss func batch {batch_nb}")

        batch_size = self.options.Training.batch_size
        if mode == "classifier":
            source_time = torch.zeros(self.options.Training.batch_size, 1).to(self.device)
            alpha = torch.ones_like(source_time).view(self.options.Training.batch_size)
        else:
            source_time = torch.rand(self.options.Training.batch_size, 1).to(self.device)
            _, alpha, _ = self.diffusion_sampler.get_logsnr_alpha_sigma(source_time, (self.options.Training.batch_size))

        outputs = self.forward(batch.sources, source_time, batch.num_sequential_vectors, batch.regression_targets, mode)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        total_loss = []

        nProcess = float(len(self.process_names))
        for process in self.process_names:
            # print(f"Rank {self.global_rank}: Processing batch {batch_nb} for {process}")
            symmetric_losses, best_indices = self.symmetric_losses(
                outputs.assignments[process],
                outputs.detections[process],
                batch.assignment_targets[process],
                process
            )

            # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
            permutations = self.event_permutation_tensor[process][best_indices].T
            masks = torch.stack([target.mask for target in batch.assignment_targets[process]])
            masks = torch.gather(masks, 0, permutations)

            # ===================================================================================================
            # Balance the loss based on the distribution of various classes in the dataset.
            # ---------------------------------------------------------------------------------------------------

            # Default unity weight on correct device.
            weights = torch.ones_like(symmetric_losses)  # [:, :, batch_size]

            # Balance based on the particles present - only used in partial event training
            if self.balance_particles:
                class_indices = (masks * self.particle_index_tensor[process].unsqueeze(1)).sum(0)
                weights *= self.particle_weights_tensor[process][class_indices]

            # Balance based on the number of jets in this event
            if self.balance_jets:
                weights *= self.jet_weights_tensor[batch.num_vectors]

            weights_alpha = (alpha ** 2).unsqueeze(0).unsqueeze(0)  # [1, 1, batch_size]
            weights = weights * weights_alpha

            # Take the weighted average of the symmetric loss terms.
            masks = masks.unsqueeze(1)

            symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(masks.sum(-1), 1, None)

            assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)

            # ===================================================================================================
            # Some basic logging
            # ---------------------------------------------------------------------------------------------------
            with torch.no_grad():
                if (mode == 'classifier'):
                    for name, l in zip(self.event_info.assignments_name[process], assignment_loss):
                        self.log(f"{status}/loss/{mode}/{process}/{name}/assignment_loss", l, sync_dist=True)

                    for name, l in zip(self.event_info.assignments_name[process], detection_loss):
                        self.log(f"{status}/loss/{mode}/{process}/{name}/detection_loss", l, sync_dist=True)

                    if torch.isnan(assignment_loss).any():
                        raise ValueError(f"{process}: Assignment loss has diverged!")

                    if torch.isinf(assignment_loss).any():
                        raise ValueError(f"{process}: Assignment targets contain a collision.")
            # ===================================================================================================
            # Start constructing the list of all computed loss terms.
            # ---------------------------------------------------------------------------------------------------

            if (self.options.Training.assignment_loss_scale > 0) and (mode == 'classifier'):
                total_loss.append(assignment_loss / nProcess)

            if (self.options.Training.detection_loss_scale > 0) and (mode == 'classifier'):
                total_loss.append(detection_loss / nProcess)

            # ===================================================================================================
            # Auxiliary loss terms which are added to reconstruction loss for alternative targets.
            # ---------------------------------------------------------------------------------------------------
            if (self.options.Training.kl_loss_scale > 0) and (mode == 'classifier'):
                total_loss = self.add_kl_loss(total_loss, outputs.assignments, masks, weights, mode, process, status)
            # print(f"Rank {self.global_rank}: Processing batch {batch_nb} for {process} finished")

        batch_weights = (alpha ** 2).view(batch_size, 1)
        if self.options.Training.regression_loss_scale > 0:
            total_loss = self.add_regression_loss(total_loss, outputs.regressions, batch.regression_targets,
                                                  batch_weights, mode, status)
        # print(f"Rank {self.global_rank}: Processing batch {batch_nb} for regressions")

        if self.options.Training.classification_loss_scale > 0:
            total_loss = self.add_classification_loss(total_loss, outputs.classifications, batch.classification_targets,
                                                      batch_weights, mode, status)
        # print(f"Rank {self.global_rank}: Processing batch {batch_nb} for classification")

        if (self.options.Training.generation_loss_scale > 0 and not (mode == 'classifier')):
            total_loss = self.add_generation_loss(total_loss, outputs.pred_score["Global"],
                                                  outputs.true_score["Global"], "Global", status)
            total_loss = self.add_generation_loss(total_loss, outputs.pred_score["Sequential"],
                                                  outputs.true_score["Sequential"], "Sequential", status)

        if (self.options.Training.feature_generation_loss_scale > 0 and not (mode == 'classifier')):
            total_loss = self.add_generation_loss(total_loss, outputs.pred_score["Feature"],
                                                  outputs.true_score["Feature"], "Feature", status)

        # print(f"Rank {self.global_rank}: Processing batch {batch_nb} return loss.")

        return total_loss
