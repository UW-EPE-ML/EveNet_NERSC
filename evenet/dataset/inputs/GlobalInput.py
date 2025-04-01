from typing import Tuple

import h5py
import numpy as np

import torch
from torch import Tensor

from omninet.dataset.types import SpecialKey, Statistics, Source, IndexDict
from omninet.dataset.inputs.BaseInput import BaseInput

class GlobalInput(BaseInput):

    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        input_group = [SpecialKey.Inputs, self.input_name]
        num_events   = len(limit_index)

        # Try and load a mask for this global features. If none is present, assume all vectors are valid.
        try:
            source_mask = torch.from_numpy(self.dataset(hdf5_file, input_group, SpecialKey.Mask)[limit_index]).contiguous()
        except KeyError:
            source_mask = torch.ones(num_events, dtype=torch.bool)

        # Load in vector features.
        num_features = self.event_info.num_features(self.input_name)
        source_data = torch.empty(num_features, num_events, dtype=torch.float32)

        for index, (feature, _, log_transform) in enumerate(self.event_info.input_features[self.input_name]):
            self.dataset(hdf5_file, input_group, feature).read_direct(source_data[index].numpy(), source_sel = limit_index)
            if log_transform:
                source_data[index] += 1
                torch.log_(source_data[index])
                source_data[index] *= source_mask
        # Reshape and limit data to the limiting index.
        source_data = source_data.transpose(0, 1)
        # Add a fake timestep dimension to global vectors.
        # noinspection PyAttributeOutsideInit
        self.source_data = source_data.unsqueeze(1)
        # noinspection PyAttributeOutsideInit
        self.source_mask = source_mask.unsqueeze(1)

    @property
    def reconstructable(self) -> bool:
        return False

    # noinspection PyAttributeOutsideInit
    def limit(self, event_mask):
        self.source_data = self.source_data[event_mask].contiguous()
        self.source_mask = self.source_mask[event_mask].contiguous()

    def compute_statistics(self) -> Tuple[Tensor, Tensor]:
        masked_data = self.source_data[self.source_mask]
        masked_mean = masked_data.mean(0)
        masked_std = masked_data.std(0)

        masked_std[masked_std < 1e-5] = 1

        masked_mean[~self.event_info.normalized_features(self.input_name)] = 0
        masked_std[~self.event_info.normalized_features(self.input_name)] = 1

        return Statistics(masked_mean, masked_std)

    def num_vectors(self) -> int:
        return self.source_mask.sum(1)

    def max_vectors(self) -> int:
        return self.source_mask.shape[1]

    def data(self):
        return Source(self.source_data, self.source_mask)

    def __getitem__(self, item):
        return Source(self.source_data[item], self.source_mask[item])
