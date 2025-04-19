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
