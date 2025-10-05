from copy import deepcopy
from collections import OrderedDict

from evenet.control.global_config import global_config

global_config.load_yaml("/Users/avencastmini/PycharmProjects/EveNet/share/finetune-example.yaml")

a = global_config.to_dict()

global_config.load_yaml("/Users/avencastmini/PycharmProjects/EveNet/share/finetune-example-2.yaml")

b = global_config.to_dict()

import torch
import numpy as np


def safe_equal(v1, v2):
    # identity or direct scalar equality
    if v1 is v2:
        return True
    if type(v1) != type(v2):
        return False

    # handle None
    if v1 is None or v2 is None:
        return v1 == v2

    # handle tensors
    if isinstance(v1, torch.Tensor):
        if v1.shape != v2.shape:
            return False
        if v1.numel() == 1:
            return torch.equal(v1, v2)
        return torch.allclose(v1, v2)

    # handle numpy arrays
    if isinstance(v1, np.ndarray):
        if v1.shape != v2.shape:
            return False
        return np.allclose(v1, v2)

    # handle OrderedDict or dict
    if isinstance(v1, (dict, OrderedDict)):
        if v1.keys() != v2.keys():
            return False
        for k in v1:
            if not safe_equal(v1[k], v2[k]):
                return False
        return True

    # handle list or tuple
    if isinstance(v1, (list, tuple)):
        if len(v1) != len(v2):
            return False
        return all(safe_equal(a, b) for a, b in zip(v1, v2))

    # fallback scalar comparison
    try:
        return v1 == v2
    except Exception:
        return False

def compare_attributes(obj1, obj2):
    diffs = {}
    for key in vars(obj1).keys() | vars(obj2).keys():
        print(key)
        v1 = getattr(obj1, key, None)
        v2 = getattr(obj2, key, None)
        try:
            if not safe_equal(v1, v2):
                diffs[key] = (v1, v2)
        except Exception as e:
            # skip if not comparable (e.g. complex structures)
            diffs[key] = f"<uncomparable: {e}>"
    return diffs

d = compare_attributes(a['event_info'], b['event_info'])

pass