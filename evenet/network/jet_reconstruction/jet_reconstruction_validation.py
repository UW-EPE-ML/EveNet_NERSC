from typing import Dict, Callable
from evenet.dataset.types import Batch, Source
from torch import Tensor
import warnings

import numpy as np
import torch

from sklearn import metrics as sk_metrics
import seaborn as sn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
# from pretty_confusion_matrix import pp_matrix

from evenet.control.config import Config
from evenet.dataset.evaluator import SymmetricEvaluator
from evenet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
import matplotlib.pyplot as plt
import os

import wandb

class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, config: Config, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(config, torch_script)
        
        # evaluator is a dictionary with process name as key
        self.evaluator = {process: SymmetricEvaluator(self.event_info, process) for process in self.process_names}
        self.confusion_matrix = dict()

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": lambda t, p: sk_metrics.recall_score(t,  p,  zero_division=0),
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p, zero_division=0),
            "f_score":     lambda t, p: sk_metrics.f1_score(t, p, zero_division=0)
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks, process, sources=None):
        event_permutation_group = self.event_permutation_tensor[process].cpu().numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.
        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        for i, permutation in enumerate(event_permutation_group):
            for j, (prediction, target) in enumerate(zip(jet_predictions, stacked_targets[permutation])):
                jet_accuracies[i, j] = np.all(prediction == target, axis=1)

            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        jet_accuracies = jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[process][jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        jet_accuracies = jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)



        # print(f"sources: {sources}")
        # raise Exception("uuuu")


        # compute mass by loop over targets
        resonance_masses = dict()
        if sources is not None:

            # check sequential data!!!
            source_data = sources[0][0] 

            # loop over targets
            for t in range(num_targets):
                event_particle_name = self.event_info.event_particles[process].names[t]
                permuted_jet_pred = torch.tensor(jet_predictions[t], dtype=torch.int32)

                # get 4-vector of all jets
                jets_mass = torch.exp(source_data[:,:,0]) # (num_of_evts, num_of_jets)
                jets_pt = torch.exp(source_data[:,:,1])
                
                jets_eta = source_data[:,:,2]
                jets_phi = source_data[:,:,3]

                # get the 4-vector of predicted jets
                jets_mass = jets_mass.to(permuted_jet_pred.device) # move data to same device (cpu or gpu)
                jets_pt = jets_pt.to(permuted_jet_pred.device)
                jets_eta = jets_eta.to(permuted_jet_pred.device)
                jets_phi = jets_phi.to(permuted_jet_pred.device)

                jets_phi_pred = jets_phi[permuted_jet_pred]

                # make indices to get predict the values
                row_indices = torch.arange(jets_phi.size(0)).unsqueeze(1).expand(-1, jets_phi_pred.size(1))  # Shape: [64, 3]
                column_indices = permuted_jet_pred  # Shape: [64, 3]


                # get values from jets_* using the batch indices and column indices
                jets_mass_pred = jets_mass[row_indices, column_indices]
                jets_pt_pred = jets_pt[row_indices, column_indices]
                jets_eta_pred = jets_eta[row_indices, column_indices]
                jets_phi_pred = jets_phi[row_indices, column_indices]

                # reconstruct four-momentum of jets
                jets_px = jets_pt_pred * torch.cos(jets_phi_pred)
                jets_py = jets_pt_pred * torch.sin(jets_phi_pred)
                jets_pz = jets_pt_pred * torch.sinh(jets_eta_pred)
                jets_energy = torch.sqrt(jets_mass_pred**2 + jets_px**2 + jets_py**2 + jets_pz**2)
                jets_four_momentum = torch.stack( (jets_energy, jets_px, jets_py, jets_pz), dim=-1) # (num_events, num_jets, 4)

                # reconstruct four-momentum of resonance
                resonance_four_momentum = jets_four_momentum.sum(dim=1) # (num_events, 1, 4)
                resonance_energy = resonance_four_momentum[..., 0]
                resonance_px = resonance_four_momentum[..., 1]
                resonance_py = resonance_four_momentum[..., 2]
                resonance_pz = resonance_four_momentum[..., 3]

                # resonance_mass^2 = E^2 - p^2
                resonance_mass = torch.sqrt(resonance_energy**2 - resonance_px**2 - resonance_py**2 - resonance_pz**2) # (num_targets, num_events)
                resonance_mass = torch.nan_to_num(resonance_mass, nan = 0.0)[stacked_masks[t]]

                # print(f"mass: {resonance_mass.shape}") 
                # print(f"mass: {resonance_mass}") 
                if torch.isnan(resonance_mass).any():
                    print(f"nan in resonance_mass: {resonance_masses}")
                    raise Exception("Done")

                resonance_masses[event_particle_name] = resonance_mass






        # Create the logging dictionaries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
    
            metrics = {f"{process}/jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                    for j in range(1, num_targets + 1)
                    for i in range(1, j + 1)}

            metrics.update({f"{process}/particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                            for j in range(1, num_targets + 1)
                            for i in range(1, j + 1)})
            
            metrics.update({f"{process}/mass_histogram/{event_particle_name}_mass": resonance_masses[event_particle_name]
                            for event_particle_name in resonance_masses})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        for name, metric in self.particle_metrics.items():
            metrics[f"{process}/particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"{process}/particle/{name}"] = metric(particle_targets, particle_scores)

        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        metrics[f"{process}/validation_accuracy"] = metrics[f"{process}/jet/accuracy_{num_targets}_of_{num_targets}"]

        return metrics

    def on_validation_epoch_start(self):
        # declare dict to accumlate mass from all validation step
        self.aggregated_metrics = {}

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        # Run the base prediction step
        sources, num_jets, targets, regression_targets, classification_targets, num_seq_jets = batch

        batch_size = num_jets.shape[0]

        # Validation Loss
        total_loss = []
      
        with torch.no_grad(): 
            total_loss += self.loss_func(batch, batch_idx, "classifier", "validation")
            if (self.options.Training.generation_loss_scale > 0 or self.options.Training.feature_generation_loss_scale > 0):
                total_loss += self.loss_func(batch, batch_idx, "generator", "validation")
            # ===================================================================================================
            # Combine and return the loss
            # ---------------------------------------------------------------------------------------------------
            total_loss = torch.cat([loss.view(-1) for loss in total_loss])

            self.log("validation/loss/total_loss", total_loss.sum(), sync_dist=True)


        source_time = torch.rand(batch_size, 1).to(self.device)

        jet_predictions, particle_scores, regressions, classifications = self.predict(sources, source_time, num_seq_jets)
        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.

        validation_accuracy = []
        metrics = dict()
        for process in self.process_names:
            num_targets = len(targets[process])
            stacked_targets = np.zeros(num_targets, dtype=object)
            stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
            for i, (target, mask) in enumerate(targets[process]):
                stacked_targets[i] = target.detach().cpu().numpy()
                stacked_masks[i] = mask.detach().cpu().numpy()
            metrics.update(self.evaluator[process].full_report_string(jet_predictions[process], stacked_targets, stacked_masks, prefix=f"{process}/Purity/"))



            # Apply permutation groups for each target
            for target, prediction, permutation_indices in zip(stacked_targets, jet_predictions[process], self.permutation_indices[process]):
                for indices in permutation_indices:
                    if len(indices) > 1:
                        prediction[:, indices] = np.sort(prediction[:, indices])
                        target[:, indices] = np.sort(target[:, indices])

            metrics.update(self.compute_metrics(jet_predictions[process], particle_scores[process], stacked_targets, stacked_masks, process, sources))
            validation_accuracy.append(metrics[f"{process}/validation_accuracy"])
        metrics["validation_accuracy"] = np.mean(validation_accuracy)

        regression_masks = {
            key: value.mask.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }


        regression_targets = {
            key: value.data.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }


        classification_targets = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }


        for key in regressions:
            regressions[key] = regressions[key][regression_masks[key]]
            regression_targets[key] = regression_targets[key][regression_masks[key]]
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            if len(percent_error) > 0:
                self.log(f"REGRESSION/{key}_percent_error", np.nanmean(percent_error), sync_dist=True)
            else:
                self.log(f"REGRESSION/{key}_percent_error", float('nan'), sync_dist=True)

            absolute_error = np.abs(delta)
            if len(absolute_error) > 0:
                self.log(f"REGRESSION/{key}_absolute_error", np.nanmean(absolute_error), sync_dist=True)
            else:
                self.log(f"REGRESSION/{key}_absolute_error", float('nan'), sync_dist=True)

            percent_deviation = delta / regression_targets[key]
            #self.logger.experiment.add_histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, self.global_step) # TensorBoard
            #percent_deviation = wandb.plot.histogram(np.array(percent_deviation), "percent deviation")
            #self.logger.experiment.log(f"REGRESSION/{key}_percent_deviation", percent_deviation)

            #absolute_deviation = delta
            #self.logger.experiment.add_histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, self.global_step)
        if len(classifications) > 0:
          for key in classifications:
            accuracy = (classifications[key] == classification_targets[key])
            self.log(f"CLASSIFICATION/{key}_accuracy", accuracy.mean(), sync_dist=True)
            key_first = key.split('/')[0]
            key_second = key.split('/')[1]
            class_label = self.event_info.class_label[key_first][key_second][0]
            confusion_Matrix = confusion_matrix(classification_targets[key], classifications[key], labels = range(len(class_label)))
            if batch_idx == 0:
                self.confusion_matrix[key] = confusion_Matrix
            else:
                self.confusion_matrix[key] += confusion_Matrix

            if (batch_idx == (self.trainer.num_val_batches[0] - 1)): 
                df_cm = pd.DataFrame(self.confusion_matrix[key], index = class_label, columns = class_label).astype('int')
                # pp_matrix(df_cm, annot=True, fmt = '.0f', cmap='summer')
                self.logger.experiment.log({f"CLASSIFICATION/{key}_validation_confusion_matrix": wandb.Image(plt)}, commit=False)
                plt.close() 

        for name, value in metrics.items():
            # if the data is tensor (like resonance mass data)
            if isinstance(value, torch.Tensor):
                # flatten and convert tensor to numpy 
                flat_data = value.cpu().numpy().flatten()
                nan_mask = np.isnan(flat_data)
                if nan_mask.any():
                    flat_data[nan_mask] = np.nanmean(flat_data) 
                if flat_data.size > 0:
                    self.log(f"{name}_mean", np.nanmean(flat_data), sync_dist=True)
                    self.log(f"{name}_std", np.nanstd(flat_data), sync_dist=True)
                else:
                    self.log(f"{name}_mean", float('nan'), sync_dist=True)
                    self.log(f"{name}_std", float('nan'), sync_dist=True)
                # accumulate data for plotting
                if name not in self.aggregated_metrics:
                    self.aggregated_metrics[name] = []
                self.aggregated_metrics[name].extend(flat_data)
            elif not np.isnan(value):
                self.log(name, value, sync_dist=True)
        return metrics

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and self.current_epoch % 5 == 0:  # plot every 10 epochs
            for name, data in self.aggregated_metrics.items():
               # Convert to numpy array for processing
                aggregated_data = np.array(data)

                # make sure not empty
                if aggregated_data.size == 0:
                    continue
                
                # Plot the aggregated data
                plt.figure()
                plt.hist(aggregated_data, bins=50, range=(min(aggregated_data), 800))
                plt.title(name)
                plt.xlabel("Mass (GeV/c^2)")
                plt.ylabel("Counts")
                
                # Save the plot to W&B
                self.logger.experiment.log({f"{name}_aggregated": wandb.Image(plt)}, commit=False)
                plt.close()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
