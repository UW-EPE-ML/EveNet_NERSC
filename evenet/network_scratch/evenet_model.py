import lightning as pl
import torch
from pandas.io import pickle

from evenet.control.config import DotDict

from evenet.network_scratch.body.normalizer import normalizer
from torch import Tensor
from typing import Dict

class EvenetModel():
    def __init__(self, config: DotDict):
        super().__init__()
        # Initialize the model with the given configuration
        self.options = config.options
        self.event_info = config.event_info
        # self.save_hyperparameters(self.options)

        with open(self.options.normalization_file, 'rb') as f:
            loaded_normalization_dict = pickle.load(f)

        # Initialize the normalization layer
        input_normalizers_setting = {}
        for input_name, input_type in self.event_info.input_types.items():
            input_normalizers_setting_local ={
                "log_mask" : torch.tensor([feature_info.logscale for feature_info in self.event_info.input_features[input_name]], dtype = torch.int),
                "mean": torch.tensor(loaded_normalization_dict["input_mean"], dtype = torch.float)
            }

            if input_type in input_normalizers_setting:
                for element in input_normalizers_setting[input_type]:
                    input_normalizers_setting[input_type][element] = torch.cat(
                        input_normalizers_setting[input_type][element],
                        input_normalizers_setting_local[element],
                    )
            else:
                input_normalizers_setting[input_type] = input_normalizers_setting_local
        self.normalizer = normalizer(
            log_mask= input_normalizers_setting["SEQUENTIAL"]["log_mask"],
            mean = input_normalizers_setting["SEQUENTIAL"]["mean"],
            std = input_normalizers_setting["SEQUENTIAL"]["std"]
        )



    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """

        :param x:
            - x['x']: point cloud, shape (batch_size, num_objects, num_features)
            - x['x_mask']: Mask for point cloud, shape (batch_size, num_objects)
                - 1: valid point
                - 0: invalid point
            - x['conditions']: conditions, shape (batch_size, num_conditions)
            - x['conditions_mask']: Mask for conditions, shape (batch_size, num_conditions)
                - 1: valid condition
                - 0: invalid condition
            - x['classification']: classification targets, shape (batch_size,)
            - x['regression']: regression targets, shape (batch_size, num_regression_targets)
            - x['regression_mask']: Mask for regression targets, shape (batch_size, num_regression_targets)
                - 1: valid regression target
                - 0: invalid regression target
            - x['num_vectors']: number of vectors in the batch, shape (batch_size,)
            - x['num_sequential_vectors']: number of sequential vectors in the batch, shape (batch_size,)
            - x['assignment_indices']: assignment indices, shape (batch_size, num_resonaces, num_targets)
            - x['assignment_indices_mask']: Mask for assignment indices, shape (batch_size, num_resonances)
                - True: valid assignment index
                - False: invalid assignment index
            - x['assignment_mask']: assignment mask, shape (batch_size, num_resonances)
                - 1: valid assignment
                - 0: invalid assignment
        """

        # Normalize the input data
        # Embedding


