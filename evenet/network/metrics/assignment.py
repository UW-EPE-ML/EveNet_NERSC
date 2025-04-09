import re

import torch.nn

from evenet.network.loss.assignment import convert_target_assignment
from evenet.utilities.group_theory import complete_indices, symmetry_group
from evenet.control.event_info import EventInfo
from evenet.network.metrics.predict_assignment import extract_predictions

from typing import List, Dict
from torch import Tensor

from functools import reduce
from itertools import permutations, product
import warnings
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import wandb


def reconstruct_mass_peak(Jet, assignment_indices, padding_mask, log_mass=True):
    """
    *** input Jet with log pt and log mass ***
    Jet: [batch_size, num_jets, 4]
    assignment_indices: [batch_size, num_targets]
    """
    jet_pt = Jet[..., 0]
    jet_eta = Jet[..., 1]
    jet_phi = Jet[..., 2]
    jet_mass = Jet[..., 3]

    if log_mass:
        jet_pt = torch.exp(jet_pt)
        jet_mass = torch.exp(jet_mass)

    def gather_jets(jet_tensor):
        return torch.gather(jet_tensor.unsqueeze(1), 2, assignment_indices.unsqueeze(1)).squeeze(1)

    pt = gather_jets(jet_pt)
    eta = gather_jets(jet_eta)
    phi = gather_jets(jet_phi)
    mass = gather_jets(jet_mass)

    selected_mask = torch.gather(padding_mask.unsqueeze(1), 2, assignment_indices.unsqueeze(1)).squeeze(1)
    is_valid_event = selected_mask.all(dim=1)

    # 4-vector components
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + mass ** 2)

    total_e = E.sum(dim=1)
    total_px = px.sum(dim=1)
    total_py = py.sum(dim=1)
    total_pz = pz.sum(dim=1)

    mass_squared = total_e ** 2 - (total_px ** 2 + total_py ** 2 + total_pz ** 2)
    mass_squared = torch.clamp(mass_squared, min=0.0)
    invariant_mass = torch.sqrt(mass_squared)

    invariant_mass[~is_valid_event] = float(-999)
    return invariant_mass


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
        'loss': {
            'num_targets': num_targets,
            'event_particles': event_particles,
            'event_permutations': event_info.event_permutations,
        },
        'step': {
            'num_targets': num_targets,
            'event_particles': event_particles,
            'event_permutations': event_info.event_permutations,
            'product_symbolic_groups': event_info.product_symbolic_groups,
        }
    }


def predict(assignments: List[Tensor],
            detections: List[Tensor],
            product_symbolic_groups,
            event_permutations):
    device = assignments[0].device
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
                final_assignments_indices.append(torch.tensor(assignment_sorted[iorder]).to(device))
                final_assignments_probabilities.append(torch.tensor(assignment_probability[iorder]).to(device))
                detections_probabilities = 1 - (detection_prob[:, iorder] / init_probabilities)
                init_probabilities = detections_probabilities
                final_detections_probabilities.append(torch.tensor(detections_probabilities).to(device))

    return {
        "best_indices": final_assignments_indices,
        "assignment_probabilities": final_assignments_probabilities,
        "detection_probabilities": final_detections_probabilities
    }


class SingleProcessAssignmentMetrics:
    def __init__(
            self,
            device,
            event_permutations,
            event_symbolic_group,
            event_particles,
            product_symbolic_groups,
            ptetaphimass_index,
            process,
            detection_WP=[0.0, 0.5, 0.8],
            hist_xmin=0,
            hist_xmax=250,
            num_bins=125
    ):

        self.device = device
        self.event_permutations = event_permutations
        self.event_particles = event_particles
        self.event_group = event_symbolic_group
        self.target_groups = product_symbolic_groups
        self.hist_xmin = hist_xmin
        self.hist_xmax = hist_xmax
        self.num_bins = num_bins
        self.detection_WP = detection_WP
        self.ptetaphimass_index = ptetaphimass_index
        self.process = process
        clusters = []
        cluster_groups = []

        for orbit in self.event_group.orbits():
            orbit = tuple(sorted(orbit))
            names = [self.event_particles[i] for i in orbit]

            names_clean = [name.replace('/', '') for name in names]

            cluster_name = map(dict.fromkeys, names_clean)
            cluster_name = map(lambda x: x.keys(), cluster_name)
            cluster_name = ''.join(reduce(lambda x, y: x & y, cluster_name))
            clusters.append((cluster_name, names, orbit))  # ['t', ['t1', 't2'], Orbit]

            cluster_group = self.target_groups[names[0]]
            for name in names:
                assert self.target_groups[name] == cluster_group, (
                    f"Invalid symmetry group for '{name}': expected {self.target_groups[name]}, "
                    f"but got {cluster_group}."
                )

            cluster_groups.append((cluster_name, names, cluster_group))  # ['t', ['t1', 't2'], Group]

        self.clusters = clusters
        self.cluster_groups = cluster_groups

        self.bins = np.linspace(self.hist_xmin, self.hist_xmax, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])

        self.mass_spectrum = dict({
            f"{i + 1}{cluster_name}": np.zeros(self.num_bins)
            for cluster_name, particle_name, orbit in self.clusters
            for i in range(len(particle_name))
        })

        self.predict_mass_spectrum_correct = dict({
            f"{i + 1}{cluster_name}": np.zeros(self.num_bins)
            for cluster_name, particle_name, orbit in self.clusters
            for i in range(len(particle_name))
        })

        self.predict_mass_spectrum_wrong = dict({
            f"{i + 1}{cluster_name}": np.zeros(self.num_bins)
            for cluster_name, particle_name, orbit in self.clusters
            for i in range(len(particle_name))
        })

        # self.predict_mass_spectrum_correct = dict({
        #     f"{i+1}{cluster_name}": {
        #         f"{detection_wp}": np.zeros(self.num_bins) for detection_wp in self.detection_WP
        #     }
        #     for cluster_name, particle_name, orbit in self.clusters
        #     for i in range(len(particle_name))
        # })

        # self.predict_mass_spectrum_wrong = dict({
        #     f"{i+1}{cluster_name}": {
        #         f"{detection_wp}": np.zeros(self.num_bins) for detection_wp in self.detection_WP
        #     }
        #     for cluster_name, particle_name, orbit in self.clusters
        #     for i in range(len(particle_name))
        # })

    def update(
            self,
            best_indices,
            assignment_probabilities,
            detection_probabilities,
            truth_indices,
            truth_masks,
            inputs,
            inputs_mask,
    ):

        best_indices, truth_indices = self.sort_outputs(best_indices, truth_indices)  # Remove intra-particle symmetries

        correct_assigned = self.check_correct_assignment(
            best_indices,
            truth_indices,
            truth_masks
        )

        for cluster_name, names, orbit in self.clusters:

            truth_count = torch.stack([truth_masks[iorbit] for iorbit in list(sorted(orbit))], dim=0).int().sum(dim=0)
            truth = torch.stack([truth_indices[iorbit] for iorbit in list(sorted(orbit))], dim=0)
            truth_masking = torch.stack([truth_masks[iorbit] for iorbit in list(sorted(orbit))], dim=0)
            prediction = torch.stack([best_indices[iorbit] for iorbit in list(sorted(orbit))], dim=0)
            predict_detection = torch.stack([detection_probabilities[iorbit] for iorbit in list(sorted(orbit))], dim=0)
            correct_reco = torch.stack([correct_assigned[iorbit] for iorbit in list(sorted(orbit))], dim=0)

            for num_resonance in range(len(names)):
                truth_mask = (truth_count == (num_resonance + 1))
                hist_name = f"{num_resonance + 1}{cluster_name}"
                for local_resonance in range(len(names)):
                    truth_local = truth[local_resonance, :, :]
                    truth_mask_local = truth_mask & truth_masking[local_resonance, :]
                    truth_local = truth_local[truth_mask_local]
                    if not (truth_local.size()[0] > 0):
                        continue

                    input = inputs[truth_mask_local]
                    input_mask = inputs_mask[truth_mask_local]
                    jet = input[:, :, self.ptetaphimass_index]
                    truth_mass = reconstruct_mass_peak(jet, truth_local, input_mask)
                    truth_mass = truth_mass.detach().cpu().numpy()
                    hist, _ = np.histogram(truth_mass, bins=self.bins)
                    self.mass_spectrum[hist_name] += hist

                    prediction_local = prediction[local_resonance, :, :][truth_mask_local]
                    detection_local = predict_detection[local_resonance, :][truth_mask_local]
                    correct_local = correct_reco[local_resonance, :][truth_mask_local]

                    predict_correct = prediction_local[correct_local]
                    detection_correct = detection_local[correct_local]
                    if prediction_local.size()[0] > 0:
                        reco_mass_correct = reconstruct_mass_peak(
                            jet[correct_local], predict_correct, input_mask[correct_local]
                        ).detach().cpu().numpy()
                        hist, _ = np.histogram(reco_mass_correct, bins=self.bins)
                        self.predict_mass_spectrum_correct[hist_name] += hist

                    prediction_false = prediction_local[~correct_local]
                    detection_false = detection_local[~correct_local]
                    if prediction_false.size()[0] > 0:
                        reco_mass_false = reconstruct_mass_peak(
                            jet[~correct_local], prediction_false, input_mask[~correct_local]
                        ).detach().cpu().numpy()
                        hist, _ = np.histogram(reco_mass_false, bins=self.bins)
                        self.predict_mass_spectrum_wrong[hist_name] += hist

    def check_correct_assignment(
            self,
            prediction,
            target_indices,
            target_masks,
    ):

        result = [torch.zeros_like(prediction[i]).bool() for i in range(len(prediction))]
        for cluster_name, cluster_particles, cluster_indices in self.clusters:
            cluster_target_masks = torch.stack([target_masks[i] for i in cluster_indices])
            cluster_target_indices = torch.stack([target_indices[i] for i in cluster_indices])
            cluster_predictions = torch.stack([prediction[i] for i in cluster_indices])
            correct_predictions = torch.zeros_like(cluster_target_masks, dtype=torch.int64)
            for target_permutation in permutations(range(len(cluster_indices))):
                target_permutation = torch.tensor(
                    target_permutation, dtype=torch.int64,
                    device=cluster_target_masks.device
                )
                prediction_correct = (cluster_predictions == cluster_target_indices[target_permutation])

                prediction_correct = prediction_correct.all(-1) * cluster_target_masks[target_permutation]
                correct_predictions = torch.maximum(prediction_correct, correct_predictions)

            for ilocal, iglobal in enumerate(cluster_indices):
                result[iglobal] = correct_predictions[ilocal, :].bool()

        return result

    def reset(self):
        for name, hist in self.mass_spectrum.items():
            self.mass_spectrum[name] = np.zeros(self.num_bins)
        for name, hist in self.predict_mass_spectrum_correct.items():
            self.predict_mass_spectrum_correct[name] = np.zeros(self.num_bins)
        for name, hist in self.predict_mass_spectrum_wrong.items():
            self.predict_mass_spectrum_wrong[name] = np.zeros(self.num_bins)

    def reduce_across_gpus(self):
        if torch.distributed.is_initialized():
            for name, hist in self.mass_spectrum.items():
                tensor = torch.tensor(hist, dtype=torch.long, device=self.device)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                self.mass_spectrum[name] = tensor.cpu().numpy()
            for name, hist in self.predict_mass_spectrum_correct.items():
                tensor = torch.tensor(hist, dtype=torch.long, device=self.device)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                self.predict_mass_spectrum_correct[name] = tensor.cpu().numpy()
            for name, hist in self.predict_mass_spectrum_wrong.items():
                tensor = torch.tensor(hist, dtype=torch.long, device=self.device)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                self.predict_mass_spectrum_wrong[name] = tensor.cpu().numpy()

    def plot_mass_spectrum_func(self,
                                truth,
                                predict_correct,
                                predict_wrong,
                                ):

        fig, ax = plt.subplots(figsize=(9, 6))
        base_colors = plt.cm.Set2(np.linspace(0.2, 0.8, 2))
        lighter = lambda c: tuple(min(1.0, x + 0.3) for x in c)

        total_pred = np.zeros_like(predict_wrong)

        ax.bar(
            self.bin_centers,
            predict_correct,
            width=np.diff(self.bins),
            bottom=total_pred,
            color=base_colors[0],
            alpha=0.6,
            label='Reco Success',
        )

        ax.bar(
            self.bin_centers,
            predict_wrong,
            width=np.diff(self.bins),
            bottom=total_pred + predict_correct,
            color=lighter(base_colors[0]),
            alpha=0.5,
            label='Reco False'
        )

        ax.step(self.bin_centers, truth, where='mid', color='black', linewidth=1.5, label='Truth')

        def gauss(x, a, mu, sigma):
            return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        try:
            popt_truth, _ = curve_fit(gauss, self.bin_centers, truth, p0=[np.max(truth), 100, 10])
            ax.plot(self.bin_centers, gauss(self.bin_centers, *popt_truth), 'k--',
                    label=f'Truth Fit: μ={popt_truth[1]:.2f}, σ={popt_truth[2]:.2f}')
        except RuntimeError:
            print("Truth fit failed")

        try:
            popt_pred, _ = curve_fit(gauss, self.bin_centers, predict_correct + predict_wrong,
                                     p0=[np.max(predict_correct + predict_wrong), 100, 10])
            ax.plot(self.bin_centers, gauss(self.bin_centers, *popt_pred), 'r--',
                    label=f'Pred Fit: μ={popt_pred[1]:.2f}, σ={popt_pred[2]:.2f}')
        except RuntimeError:
            print("Prediction fit failed")

        ax.legend()
        fig.tight_layout()
        return fig

    def plot_mass_spectrum(self):
        return_plot = dict()
        for name, hist in self.mass_spectrum.items():
            return_plot[f"{name}"] = self.plot_mass_spectrum_func(
                hist,
                self.predict_mass_spectrum_correct[name],
                self.predict_mass_spectrum_wrong[name],
            )
        return return_plot

    @staticmethod
    def permute_arrays(self, array_list, permutation):
        return [array_list[index] for index in permutation]

    def sort_outputs(self, predictions, targets):
        """
        :param predictions:
        :param targets:
        :return:
        Sort all of the targets and predictions to avoid any intra-particle symmetries
        """

        predictions = [torch.clone(p) for p in predictions]
        targets = [torch.clone(p) for p in targets]
        for i, (_, particle_group) in enumerate(self.target_groups.items()):
            for orbit in particle_group.orbits():
                orbit = tuple(sorted(orbit))

                targets[i][:, orbit] = torch.sort(targets[i][:, orbit], dim=1)[0]
                predictions[i][:, orbit] = torch.sort(predictions[i][:, orbit], dim=1)[0]
        return predictions, targets


def shared_step(
        ass_loss_fn,
        loss_dict,
        assignment_loss_scale,
        detection_loss_scale,
        process_names,
        assignments,
        detections,
        targets,
        targets_mask,
        product_symbolic_groups,
        event_permutations,
        batch_size,
        device,
        event_particles,
        num_targets,
        point_cloud,
        point_cloud_mask,
        subprocess_id,
        metrics: dict[str, SingleProcessAssignmentMetrics]
):
    symmetric_losses = ass_loss_fn(
        assignments=assignments,
        detections=detections,
        targets=targets,
        targets_mask=targets_mask,
        process_id=subprocess_id,
    )

    assignment_predict = dict()
    ass_target_metric, ass_mask_metric = convert_target_assignment(
        targets=targets,
        targets_mask=targets_mask,
        event_particles=event_particles,
        num_targets=num_targets
    )

    assignment_loss = torch.zeros(batch_size, device=device, requires_grad=True)
    detected_loss = torch.zeros(batch_size, device=device, requires_grad=True)
    for process in process_names:
        assignment_predict[process] = predict(
            assignments=assignments[process],
            detections=detections[process],
            product_symbolic_groups=product_symbolic_groups[process],
            event_permutations=event_permutations[process],
        )

        metrics[process].update(
            best_indices=assignment_predict[process]["best_indices"],
            assignment_probabilities=assignment_predict[process]["assignment_probabilities"],
            detection_probabilities=assignment_predict[process]["detection_probabilities"],
            truth_indices=ass_target_metric[process],
            truth_masks=ass_mask_metric[process],
            inputs=point_cloud,
            inputs_mask=point_cloud_mask,
        )

        loss_dict[f"ass-{process}"] = symmetric_losses["assignment"][process]
        loss_dict[f"det-{process}"] = symmetric_losses["detection"][process]

        assignment_loss = assignment_loss + symmetric_losses["assignment"][process]
        detected_loss = detected_loss + symmetric_losses["detection"][process]

    loss_dict['assignment_loss'] = assignment_loss
    loss_dict['detection_loss'] = detected_loss

    total_loss = assignment_loss_scale * assignment_loss + detection_loss_scale * detected_loss

    return total_loss, assignment_predict


def shared_epoch_end(
        global_rank,
        metrics_valid,
        logger,
):
    for process in metrics_valid:
        metrics_valid[process].reduce_across_gpus()

    if global_rank == 0:

        for process in metrics_valid:
            figs = metrics_valid[process].plot_mass_spectrum()
            logger.log({
                f"assignment_reco_mass/{process}/{name}": wandb.Image(fig)
                for name, fig in figs.items()
            })
            for _, fig in figs.items():
                plt.close(fig)

        for _, metric in metrics_valid.items():
            metric.reset()
