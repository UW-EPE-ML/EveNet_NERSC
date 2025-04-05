from evenet.control import event_info
from evenet.control.event_info import EventInfo
from collections import OrderedDict
import torch
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from typing import List, Tuple, Dict
from evenet.utilities.group_theory import complete_indices, symmetry_group
import re


def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list

    return output


def assignment_cross_entropy_loss(prediction: Tensor, target_data: Tensor, target_mask: Tensor, gamma: float) -> Tensor:
    batch_size = prediction.shape[0]
    prediction_shape = prediction.shape[1:]

    # Remove missing jets
    target_data = target_data.clamp(0, None)

    # Find the unravelling shape required to flatten the target indices
    ravel_sizes = torch.tensor(prediction_shape).flip(0)
    ravel_sizes = torch.cumprod(ravel_sizes, 0)
    ravel_sizes = torch.div(ravel_sizes, ravel_sizes[0], rounding_mode='floor')
    # ravel_sizes = ravel_sizes // ravel_sizes[0]
    ravel_sizes = ravel_sizes.flip(0).unsqueeze(0)
    ravel_sizes = ravel_sizes.to(target_data.device)

    # Flatten the target and predicted data to be one dimensional
    ravel_target = (target_data * ravel_sizes).sum(1)
    ravel_prediction = prediction.reshape(batch_size, -1).contiguous()

    log_probability = ravel_prediction.gather(-1, ravel_target.view(-1, 1)).squeeze()
    log_probability = log_probability.masked_fill(~target_mask, 0.0)

    focal_scale = (1 - torch.exp(log_probability)) ** gamma

    return -log_probability * focal_scale

    self.event_info = event_info
    self.event_permutation_tensor = OrderedDict()
    self.device = device

    self.permutation_indices = permutation_indices
    self.num_targets = num_targets

    print("configure permutation indices")


def particle_symmetric_loss(assignment: Tensor,
                            detection: Tensor,
                            target: Tensor,
                            mask: Tensor,
                            focal_gamma: float,
                            assignment_loss_scale: float,
                            detection_loss_scale: float) -> Tensor:
    assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, focal_gamma)
    detection_loss = F.binary_cross_entropy_with_logits(detection, mask.float(), reduction='none')  # TODO
    return torch.stack((
        assignment_loss_scale * assignment_loss,
        detection_loss_scale * detection_loss
    ))


def compute_symmetric_losses(assignments: List[Tensor],
                             detections: List[Tensor],
                             targets: List[Tensor],
                             process: str,
                             event_permutation_tensor: Dict,
                             focal_gamma: float,
                             assignment_loss_scale: float,
                             detection_loss_scale: float) -> Tensor:
    symmetric_losses = []
    for permutation in event_permutation_tensor[process].cpu().numpy():
        current_permutation_loss = tuple(
            particle_symmetric_loss(assignment, detection, target, mask, focal_gamma, assignment_loss_scale,
                                    detection_loss_scale)
            for assignment, detection, (target, mask)
            in zip(assignments, detections, targets[permutation])
        )
        symmetric_losses.append(torch.stack(current_permutation_loss))
    return torch.stack(symmetric_losses)


def combine_symmetric_losses(symmetric_losses: Tensor,
                             combine_pair_loss: str):
    total_symmetric_loss = symmetric_losses.sum((1, 2))
    index = total_symmetric_loss.argmin(0)
    combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]

    if combine_pair_loss.lower() == 'mean':
        combined_loss = symmetric_losses.mean(0)
    if combine_pair_loss.lower() == 'softmin':
        weights = F.softmin(total_symmetric_loss, 0)
        weights = weights.unsqueeze(1).unsqueeze(1)
        combined_loss = (weights * symmetric_losses).sum(0)

    return combined_loss, index


def symmetric_loss(assignments: List[Tensor],
                   detections: List[Tensor],
                   targets: List[Tensor],
                   process: str,
                   combine_pair_loss: str,
                   num_targets: Dict,
                   event_permutation_tensor: Dict,
                   focal_gamma: float,
                   assignment_loss_scale: float,
                   detection_loss_scale: float
                   ) -> Tuple[Tensor, Tensor]:
    symmetric_losses_summary = OrderedDict()
    assignments = [prediction + torch.log(torch.scalar_tensor(num_target))
                   for prediction, num_target in zip(assignments, num_targets[process])]
    targets = numpy_tensor_array(targets)

    symmetric_losses = compute_symmetric_losses(assignments,
                                                detections,
                                                targets,
                                                process,
                                                event_permutation_tensor,
                                                focal_gamma,
                                                assignment_loss_scale,
                                                detection_loss_scale)
    return combine_symmetric_losses(symmetric_losses, combine_pair_loss)


def loss(assignments: List[Tensor],
         detections: List[Tensor],
         targets: List[Tensor],
         process: str,
         combine_pair_loss: str,
         num_targets: Dict,
         event_permutation_tensor: Dict,
         focal_gamma: float,
         assignment_loss_scale: float,
         detection_loss_scale: float
         ):

    symmetric_losses = symmetric_loss(assignments,
                                      detections,
                                      targets,
                                      process,
                                      combine_pair_loss,
                                      num_targets,
                                      event_permutation_tensor,
                                      focal_gamma,
                                      assignment_loss_scale,
                                      detection_loss_scale)

    symmetric_losses = symmetric_losses.sum(-1)

    assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)

    return assignment_loss, detection_loss
