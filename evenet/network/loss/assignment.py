from collections import OrderedDict
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Dict


# def numpy_tensor_array(tensor_list):
#     output = np.empty(len(tensor_list), dtype=object)
#     output[:] = tensor_list[:]
#
#     return output
# def combine_symmetric_losses(symmetric_losses: Tensor,
#                              combine_pair_loss: str):
#
#     # No needed as we already incoorporated the event permutation in the model structure
#     total_symmetric_loss = symmetric_losses.sum((1, 2))
#     index = total_symmetric_loss.argmin(0)
#     combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]
#
#     if combine_pair_loss.lower() == 'mean':
#         combined_loss = symmetric_losses.mean(0)
#     if combine_pair_loss.lower() == 'softmin':
#         weights = F.softmin(total_symmetric_loss, 0)
#         weights = weights.unsqueeze(1).unsqueeze(1)
#         combined_loss = (weights * symmetric_losses).sum(0)
#
#     return combined_loss, index

def convert_target_assignment_array(
        targets: List[Tensor],
        targets_mask: List[Tensor],
        event_particles: Dict,
        num_targets: Dict
):
    """
    Convert target assignment array to a dict of tensors list.
    """
    target_assignment = OrderedDict()
    target_assignment_mask = OrderedDict()

    index_global = 0
    for process in event_particles.keys():
        target_assignment[process] = []
        target_assignment_mask[process] = []
        local_index = 0
        for event_particle in event_particles[process]:
            target_assignment[process].append(targets[:, index_global, :][..., :(num_targets[process][local_index])])
            target_assignment_mask[process].append(targets_mask[:, index_global])
            index_global += 1
            local_index += 1

    return target_assignment, target_assignment_mask


def assignment_cross_entropy_loss(prediction: Tensor, target_data: Tensor, target_mask: Tensor, gamma: float) -> Tensor:
    batch_size = prediction.shape[0]
    prediction_shape = prediction.shape[1:]

    # Remove missing jets
    target_data = target_data.clamp(0, None)

    # Find the unravelling shape required to flatten the target indices
    ravel_sizes = torch.tensor(prediction_shape).flip(0)
    ravel_sizes = torch.cumprod(ravel_sizes, 0)
    ravel_sizes = torch.div(ravel_sizes, ravel_sizes[0], rounding_mode='floor')  # [1, num_jets, num_jets * num_jets]
    # ravel_sizes = ravel_sizes // ravel_sizes[0]
    ravel_sizes = ravel_sizes.flip(0).unsqueeze(0)  # reverse, unsqueeze to add batch dimension
    ravel_sizes = ravel_sizes.to(target_data.device)

    # Flatten the target and predicted data to be one dimensional
    ravel_target = (target_data * ravel_sizes).sum(1)  # ravel_index (flatten the assignment matrix)
    ravel_prediction = prediction.reshape(batch_size, -1).contiguous()

    log_probability = ravel_prediction.gather(-1, ravel_target.view(-1, 1)).squeeze()
    log_probability = log_probability.masked_fill(~target_mask, 0.0) # 1e-6

    # focal_scale = (1 - torch.exp(log_probability)) ** gamma

    p = torch.exp(log_probability)
    # Compute focal scale only where valid
    focal_base = (1 - p).clamp(min=1e-6)
    focal_scale = torch.zeros_like(p)
    focal_scale[target_mask] = focal_base[target_mask] ** gamma

    return -log_probability * focal_scale


def compute_symmetric_losses(
        assignments: List[Tensor],
        targets: List[Tensor],
        targets_mask: List[Tensor],
        focal_gamma: float
) -> Tensor:
    # For current encoder structure, the event permutation is already embedded in the model structure, so no need for
    # specific event permutation here.

    # for permutation in event_permutation_tensor[process]:

    current_permutation_loss = tuple(
        assignment_cross_entropy_loss(assignment, target, mask, focal_gamma)
        for assignment, target, mask
        in zip(assignments, targets, targets_mask)
    )
    return torch.stack(current_permutation_loss)  # [num_particles, B]


def symmetric_loss(
        assignments: List[Tensor],
        targets: List[Tensor],
        targets_mask: List[Tensor],
        num_targets: List[int],
        focal_gamma: float,
) -> Tuple[Tensor, Tensor]:
    assignments = [
        prediction + torch.log(torch.scalar_tensor(num_target))
        for prediction, num_target in zip(assignments, num_targets)
    ]
    symmetric_losses = compute_symmetric_losses(
        assignments,
        targets,
        targets_mask,
        focal_gamma
    )
    return symmetric_losses


def loss_single_process(
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: List[Tensor],
        targets_mask: List[Tensor],
        num_targets: List[Tensor],
        event_permutations: Tuple[List],
        focal_gamma: float
):
    ####################
    ## Detection Loss ##
    ####################

    detections = detections
    detections_target = targets_mask
    detection_losses = []
    for symmetry_group in event_permutations:
        for symmetry_element in symmetry_group:
            symmetry_element = np.array(symmetry_element)
            detection = detections[symmetry_element[0]]
            detection_target = torch.stack([detections_target[symmetry_index] for symmetry_index in symmetry_element])
            detection_target = detection_target.sum(0).long()
            detection_losses.append(F.cross_entropy(
                input=detection,
                target=detection_target,
                reduction='none',
                ignore_index=-1,
            ))

    detection_loss = torch.mean(torch.stack(detection_losses))  # TODO: Check balance and masking

    #####################
    ## Assignment Loss ##
    #####################

    symmetric_losses = symmetric_loss(
        assignments,
        targets,
        targets_mask,
        num_targets,
        focal_gamma
    )

    valid_assignments = torch.sum(torch.stack(targets_mask).float())

    if valid_assignments > 0:
        assignment_loss = symmetric_losses * torch.stack(targets_mask).float()
        assignment_loss = torch.sum(assignment_loss) / valid_assignments.clamp(min=1.0)  # TODO: Check balance and masking
    else:
        assignment_loss = torch.zeros_like(valid_assignments, requires_grad=True)

    return assignment_loss, detection_loss


###################
## Main Function ##
###################

def loss(
        assignments: Dict[str, List[Tensor]],
        detections: Dict[str, List[Tensor]],
        targets: List[Tensor],
        targets_mask: List[Tensor],
        event_particles: Dict,
        event_permutations: Dict,
        num_targets: Dict,
        focal_gamma: float
):
    targets, targets_mask = convert_target_assignment_array(targets, targets_mask, event_particles, num_targets)
    loss_summary = dict({
        "assignment": dict(),
        "detection": dict()
    }
    )

    for process in event_permutations.keys():
        assignment_loss, detection_loss = loss_single_process(
            assignments[process],
            detections[process],
            targets[process],
            targets_mask[process],
            num_targets[process],
            event_permutations[process],
            focal_gamma
        )
        loss_summary["assignment"][process] = assignment_loss
        loss_summary["detection"][process] = detection_loss

    return loss_summary
