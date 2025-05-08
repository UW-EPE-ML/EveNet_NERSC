from itertools import combinations
from typing import Dict, Union, List

import numpy as np
from torch import Tensor
import torch
from collections import OrderedDict


def gather_index(x: Union[Dict, Tensor, List, None], index: Tensor):
    if x is None:
        return None
    if isinstance(x, List):
        return [element[index] for element in x]
    if isinstance(x, Tensor):
        return x[index]
    out = OrderedDict()
    for k, v in x.items():
        out[k] = gather_index(v, index)
    return out


def get_transition(global_step, start_step, duration_steps, device):
    step_tensor = torch.tensor(global_step, dtype=torch.float32, device=device)
    progress = torch.clamp((step_tensor - start_step) / duration_steps, 0.0, 1.0)
    t = 0.5 * (1 - torch.cos(np.pi * progress))
    return t


# For Torch JD
def check_param_overlap(task_param_sets, task_names, model, current_step=None, check_every=1, verbose=False):
    """
    Check and optionally list overlapping parameters between tasks.

    Args:
        task_param_sets (List[List[torch.nn.Parameter]]): List of param lists per task.
        task_names (List[str]): Task names corresponding to each param set.
        model (torch.nn.Module): Model to extract parameter names.
        current_step (int, optional): If set, only check every `check_every` steps.
        check_every (int): Frequency of checking.
        verbose (bool): If True, prints names of overlapping parameters.

    Returns:
        Dict[Tuple[str, str], Set[str]]: Overlap mapping from task pairs to parameter names.
    """
    if current_step is not None and current_step % check_every != 0:
        return {}

    # Build reverse lookup of parameter id -> name
    param_id_to_name = {id(p): n for n, p in model.named_parameters()}

    id_sets = {name: set(map(id, params)) for name, params in zip(task_names, task_param_sets)}
    overlaps = {}

    for name1, name2 in combinations(task_names, 2):
        shared_ids = id_sets[name1] & id_sets[name2]
        shared_names = {param_id_to_name[pid] for pid in shared_ids if pid in param_id_to_name}
        overlaps[(name1, name2)] = shared_names
        print(f"[TorchJD Overlap] {name1} ↔ {name2}: {len(shared_names)} shared parameters")
        if verbose and shared_names:
            for pname in sorted(shared_names):
                print(f"    ↳ {pname}")

    return overlaps
