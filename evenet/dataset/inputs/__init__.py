import h5py
import numpy as np

from omninet.dataset.event_info import EventInfo

from omninet.dataset.types import InputType
from omninet.dataset.inputs.BaseInput import BaseInput
from omninet.dataset.inputs.GlobalInput import GlobalInput
from omninet.dataset.inputs.RelativeInput import RelativeInput
from omninet.dataset.inputs.SequentialInput import SequentialInput


def create_source_input(
        event_info: EventInfo,
        hdf5_file: h5py.File,
        input_name: str,
        num_events: int,
        limit_index: np.ndarray,
) -> BaseInput:
    source_class = {
        InputType.Sequential: SequentialInput,
        InputType.Relative: RelativeInput,
        InputType.Global: GlobalInput,
    }[event_info.input_type(input_name)]

    return source_class(event_info, hdf5_file, input_name, num_events, limit_index)
