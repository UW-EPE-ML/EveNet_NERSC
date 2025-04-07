import re
from evenet.utilities.group_theory import complete_indices, symmetry_group
from evenet.control.event_info import EventInfo


def get_assignment_necessaries(
        event_info: EventInfo,
):
    permutation_indices = dict()
    num_targets = dict()
    event_permutation = dict()
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
            event_permutation[process] = complete_indices(
                event_info.event_symmetries[process].degree,
                event_info.event_symmetries[process].permutations
            )
            permutation_group = symmetry_group(permutation_indices_tmp)
            num_targets[process].append(event_info.pairing_topology_category[topology_category_name][
                    "product_symmetry"].degree)

    return {
        'num_targets': num_targets,
        'event_permutation': event_permutation,
        'event_particles': event_particles,
    }


def shared_step(
        ass_loss_fn,
        process_names,
        assignments,
        detections,
        targets,
        targets_mask,
):

    symmetric_losses = ass_loss_fn(
        assignments=assignments,
        detections=detections,
        targets=targets,
        targets_mask=targets_mask,
    )

    return symmetric_losses
