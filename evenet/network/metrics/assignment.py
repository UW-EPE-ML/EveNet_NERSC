import re

import torch.nn

from evenet.utilities.group_theory import complete_indices, symmetry_group
from evenet.control.event_info import EventInfo
from evenet.network.metrics.predict_assignment import extract_predictions

from typing import List
from torch import Tensor

from functools import reduce
from itertools import permutations, product
import warnings
import numpy as np


def get_assignment_necessaries(
        event_info: EventInfo,
):
    permutation_indices = dict()
    num_targets = dict()
    event_particles = dict()
    for process in event_info.process_names:
        permutation_indices[process] = []
        num_targets[process] = []
        for event_particle_name, product_symmetry in event_info.product_symmetries[process].items():
            topology_name = ''.join(event_info.product_particles[process][event_particle_name].names)
            topology_name = f"{event_particle_name}/{topology_name}"
            topology_name = re.sub(r'\d+', '', topology_name)
            topology_category_name = event_info.pairing_topology[topology_name][
                "pairing_topology_category"]
            permutation_indices_tmp = complete_indices(
                event_info.pairing_topology_category[topology_category_name][
                    "product_symmetry"].degree,
                event_info.pairing_topology_category[topology_category_name][
                    "product_symmetry"].permutations
            )
            permutation_indices[process].append(permutation_indices_tmp)
            event_particles[process] = [p for p in event_info.event_particles[process].names]
            num_targets[process].append(event_info.pairing_topology_category[topology_category_name][
                                            "product_symmetry"].degree)

    return {
        'loss':
            {
                'num_targets': num_targets,
                'event_particles': event_particles,
                'event_permutation': event_info.event_permutation,
            },
        'step': {
            'event_permutation': event_info.event_permutation,
            'product_symbolic_groups': event_info.product_symbolic_groups,
        }
    }


def shared_step(
        ass_loss_fn,
        loss_dict,
        loss_scale,
        process_names,
        assignments,
        detections,
        targets,
        targets_mask,
        product_symbolic_groups,
        event_permutation,
        batch_size,
        device,
):
    symmetric_losses = ass_loss_fn(
        assignments=assignments,
        detections=detections,
        targets=targets,
        targets_mask=targets_mask,
    )

    assignment_predict = dict()

    total_loss = torch.zeros(batch_size, device=device, requires_grad=True)
    for process in process_names:
        assignment_predict[process] = predict(
            assignments=assignments[process],
            detections=detections[process],
            product_symbolic_groups=product_symbolic_groups[process],
            event_permutations=event_permutation[process],
        )

        loss_dict[f"ass-{process}"] = symmetric_losses["assignment"][process]
        loss_dict[f"det-{process}"] = symmetric_losses["detections"][process]

        total_loss = total_loss + symmetric_losses["assignment"][process]
        total_loss = total_loss + symmetric_losses["detection"][process]
        total_loss = total_loss * loss_scale

    return total_loss, assignment_predict


def predict(assignments: List[Tensor],
            detections: List[Tensor],
            product_symbolic_groups,
            event_permutations):
    assignments_indices = extract_predictions(
        [
            np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
            for assignment in assignments
        ]
    )

    assignment_probabilities = []
    dummy_index = torch.arange(assignments_indices[0].shape[0])
    for assignment_probability, assignment, symmetries in zip(
            assignments,
            assignments_indices,
            product_symbolic_groups.values()
    ):
        # Get the probability of the best assignment.
        # Have to use explicit function call here to construct index dynamically.
        assignment_probability = assignment_probability.__getitem__((dummy_index, *assignment.T))
        # Convert from log-probability to probability.
        assignment_probability = torch.exp(assignment_probability)

        # Multiply by the symmetry factor to account for equivalent predictions.
        assignment_probability = symmetries.order() * assignment_probability

        # Convert back to cpu and add to database.
        assignment_probabilities.append(assignment_probability.detach().cpu().numpy())

    final_assignments_indices = []
    final_assignments_probabilities = []
    final_detections_probabilities = []
    for symmetry_group in event_permutations:
        for symmetry_element in symmetry_group:
            symmetry_element = np.sort(np.array(symmetry_element))
            detection_result = detections[symmetry_element[0]]
            softmax = torch.nn.Softmax(dim=-1)

            detection_prob = softmax(detection_result).detach().cpu().numpy()

            assignment_tmp = np.stack([assignments_indices[element] for element in symmetry_element])
            assignment_probability_tmp = np.stack([assignment_probabilities[element] for element in symmetry_element])

            sort_index = np.argsort(-1 * assignment_probability_tmp, axis=0)
            assignment_sorted = np.take_along_axis(assignment_tmp, np.expand_dims(sort_index, axis=2), axis=0)
            assignment_probability = np.take_along_axis(assignment_probability_tmp, sort_index, axis=0)

            init_probabilities = np.ones_like(assignment_probability[0])
            for iorder in range(len(symmetry_element)):
                final_assignments_indices.append(assignment_sorted[iorder])
                final_assignments_probabilities.append(assignment_probability[iorder])
                detections_probabilities = 1 - (detection_prob[:, iorder] / init_probabilities)
                init_probabilities = detections_probabilities
                final_detections_probabilities.append(detections_probabilities)

    return {
        "best_indices": final_assignments_indices,
        "assignment_probabilities": final_assignments_probabilities,
        "detection_probabilities": final_detections_probabilities
    }


class SymmetricEvaluator:
    def __init__(self, event_info: EventInfo, process: str):
        self.event_info = event_info
        self.event_group = event_info.event_symbolic_group[process]
        self.target_groups = event_info.product_symbolic_groups[process]
        self.process_name = process
        # Gather all of the Similar particles together based on the permutation groups
        clusters = []
        cluster_groups = []

        for orbit in self.event_group.orbits():
            orbit = tuple(sorted(orbit))
            names = [event_info.event_particles[process][i] for i in orbit]

            names_clean = [name.replace('/', '') for name in names]

            cluster_name = map(dict.fromkeys, names_clean)
            cluster_name = map(lambda x: x.keys(), cluster_name)
            cluster_name = ''.join(reduce(lambda x, y: x & y, cluster_name))
            clusters.append((cluster_name, names, orbit))

            cluster_group = self.target_groups[names[0]]
            for name in names:
                assert self.target_groups[name] == cluster_group, (
                    f"Invalid symmetry group for '{name}': expected {self.target_groups[name]}, "
                    f"but got {cluster_group}."
                )

            cluster_groups.append((cluster_name, names, cluster_group))

        self.clusters = clusters
        self.cluster_groups = cluster_groups

    @staticmethod
    def permute_arrays(array_list, permutation):
        return [array_list[index] for index in permutation]

    def sort_outputs(self, predictions, target_jets, target_masks):
        predictions = [np.copy(p) for p in predictions]
        target_jets = [np.copy(p) for p in target_jets]

        # Sort all of the targets and predictions to avoid any intra-particle symmetries
        for i, (_, particle_group) in enumerate(self.target_groups.items()):
            for orbit in particle_group.orbits():
                orbit = tuple(sorted(orbit))

                target_jets[i][:, orbit] = np.sort(target_jets[i][:, orbit], axis=1)
                predictions[i][:, orbit] = np.sort(predictions[i][:, orbit], axis=1)

        return predictions, target_jets, target_masks

    def particle_count_info(self, target_masks):
        target_masks = np.array(target_masks)

        # Count the total number of particles for simple filtering
        total_particle_counts = target_masks.sum(0)

        # Count the number of particles present in each cluster
        particle_counts = [
            target_masks[list(cluster_indices)].sum(0)
            for _, _, cluster_indices in self.clusters
        ]

        # Find the maximum number of particles in each cluster
        particle_max = [len(cluster_indices) for _, _, cluster_indices in self.clusters]

        return total_particle_counts, particle_counts, particle_max

    def cluster_purity(self, predictions, target_jets, target_masks):
        results = []

        for cluster_name, cluster_particles, cluster_indices in self.clusters:
            # Extract jet information for the current cluster
            cluster_target_masks = np.stack([target_masks[i] for i in cluster_indices])
            cluster_target_jets = np.stack([target_jets[i] for i in cluster_indices])
            cluster_predictions = np.stack([predictions[i] for i in cluster_indices])

            # Keep track of the best accuracy achieved for each event
            best_accuracy = np.zeros(cluster_target_masks.shape[1], dtype=np.int64)

            for target_permutation in permutations(range(len(cluster_indices))):
                target_permutation = list(target_permutation)

                accuracy = cluster_predictions == cluster_target_jets[target_permutation]
                accuracy = accuracy.all(-1) * cluster_target_masks[target_permutation]
                accuracy = accuracy.sum(0)

                best_accuracy = np.maximum(accuracy, best_accuracy)

            # Get rid of pesky warnings
            total_particles = cluster_target_masks.sum()
            if total_particles > 0:
                cluster_accuracy = best_accuracy.sum() / cluster_target_masks.sum()
            else:
                cluster_accuracy = np.nan

            results.append((cluster_name, cluster_particles, cluster_accuracy))

        return results

    def event_purity(self, predictions, target_jets, target_masks):
        target_masks = np.stack(target_masks)

        # Keep track of the best accuracy achieved for each event
        best_accuracy = np.zeros(target_masks.shape[1], dtype=np.int64)

        for target_permutation in self.event_info.event_permutation_group[self.process_name]:
            permuted_targets = self.permute_arrays(target_jets, target_permutation)
            permuted_mask = self.permute_arrays(target_masks, target_permutation)
            accuracy = np.array([
                (p == t).all(-1) * m
                for p, t, m
                in zip(predictions, permuted_targets, permuted_mask)
            ])
            accuracy = accuracy.sum(0)

            best_accuracy = np.maximum(accuracy, best_accuracy)

        # Event accuracy is defined as getting all possible particles in event
        num_particles_in_event = target_masks.sum(0)
        accurate_event = best_accuracy == num_particles_in_event

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return accurate_event.mean()

    def full_report(self, predictions, target_jets, target_masks):
        predictions, target_jets, target_masks = self.sort_outputs(predictions, target_jets, target_masks)

        total_particle_counts, particle_counts, particle_max = self.particle_count_info(target_masks)
        particle_ranges = [list(range(-1, pmax + 1)) for pmax in particle_max]

        full_results = []

        for event_counts in product(*particle_ranges):
            # Filter all events to make sure they at least have a particle there
            event_mask = total_particle_counts >= 0

            # Filter to have the correct cluster counts
            for particle_count, event_count in zip(particle_counts, event_counts):
                if event_count >= 0:
                    event_mask = event_mask & (particle_count == event_count)

                # During wildcard events, make sure we have at least one particle in the event.
                if event_count < 0:
                    event_mask = event_mask & (total_particle_counts > 0)

            # Filter event information according to computed mask
            masked_predictions = [p[event_mask] for p in predictions]
            masked_target_jets = [p[event_mask] for p in target_jets]
            masked_target_masks = [p[event_mask] for p in target_masks]

            # Compute purity values
            masked_event_purity = self.event_purity(masked_predictions, masked_target_jets, masked_target_masks)
            masked_cluster_purity = self.cluster_purity(masked_predictions, masked_target_jets, masked_target_masks)

            mask_proportion = event_mask.mean()

            full_results.append((event_counts, mask_proportion, masked_event_purity, masked_cluster_purity))

        return full_results

    def full_report_string(self, predictions, target_jets, target_masks, prefix: str = ""):
        full_purities = {}

        report = self.full_report(predictions, target_jets, target_masks)
        for event_mask, mask_proportion, event_purity, particle_purity in report:

            event_mask_name = ""
            purity = {
                "{}{}/event_purity": event_purity,
                "{}{}/event_proportion": mask_proportion
            }

            for mask_count, (cluster_name, _, cluster_purity) in zip(event_mask, particle_purity):
                mask_count = "*" if mask_count < 0 else str(mask_count)
                event_mask_name = event_mask_name + mask_count + cluster_name
                purity["{}{}/{}_purity".format("{}", "{}", cluster_name)] = cluster_purity

            purity = {
                key.format(prefix, event_mask_name): val for key, val in purity.items()
            }

            full_purities.update(purity)

        return full_purities
