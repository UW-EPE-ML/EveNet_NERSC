from typing import Dict
from collections import OrderedDict

import torch
from torch import Tensor, nn

from evenet.control.config import DotDict
from evenet.control.event_info import EventInfo
from evenet.dataset.regressions import regression_class
from evenet.network.layers.branch_linear import NormalizedBranchLinear


class RegressionDecoder(nn.Module):
    __constants__ = ['hidden_dim', 'num_layers']

    def __init__(self, options: DotDict, event_info: EventInfo, means: Dict, stds: Dict):
        super(RegressionDecoder, self).__init__()

        # Compute training dataset statistics to fix the final weight and bias.
        self.means = means
        self.stds = stds
        # A unique linear decoder for each possible regression.
        # TODO make these non-unique for symmetric indices.
        networks = OrderedDict()

        networks_param = OrderedDict()
        for name in event_info.regression_names:

            # One process one regression head
            category_name = name.split('/')[0]
            param_name = '/'.join(name.split('/')[1:])

            # Once particle one regression head
            #          category_name = '/'.join(name.split('/')[:-1])
            #          param_name    = name.split('/')[-1]
            if category_name not in networks_param:
                networks_param[category_name] = dict()
                networks_param[category_name]['params'] = [param_name]
                networks_param[category_name]['means'] = [means[name]]
                networks_param[category_name]['stds'] = [stds[name]]
                networks_param[category_name]['regression_types'] = event_info.regression_types[name]
            else:
                networks_param[category_name]['params'].append(param_name)
                networks_param[category_name]['means'].append(means[name])
                networks_param[category_name]['stds'].append(stds[name])

        print(networks_param)
        for category_name in networks_param:
            networks_param[category_name]['means'] = torch.stack(networks_param[category_name]['means'])
            networks_param[category_name]['stds'] = torch.stack(networks_param[category_name]['stds'])

            networks[category_name] = NormalizedBranchLinear(
                options,
                options.Network.num_regression_layers,
                regression_class(networks_param[category_name]['regression_types']),
                networks_param[category_name]['means'],
                networks_param[category_name]['stds']
            )

        self.networks_param = networks_param
        #        for name, data in training_dataset.regressions.items():
        #            if data is None:
        #                continue
        #            print(name)
        #            networks[name] = NormalizedBranchLinear(
        #                options,
        #                options.num_regression_layers,
        #                regression_class(training_dataset.regression_types[name]),
        #                means[name],
        #                stds[name]
        #            )

        self.networks = nn.ModuleDict(networks)

    def forward(self, vectors: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # vectors: Dict with mapping name -> [B, D]
        # outputs: Dict with mapping name -> [B, O_name]
        final_output = OrderedDict()

        for key, network in self.networks.items():
            output = network(vectors['EVENT'])
            #          print(key, output.size())
            for name_index, name in enumerate(self.networks_param[key]['params']):
                final_name = "{}/{}".format(key, name)
                final_output[final_name] = output[..., name_index]

        return final_output
