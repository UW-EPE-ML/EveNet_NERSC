from typing import Dict, Union, List
from torch import Tensor
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