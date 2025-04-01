from typing import Tuple
from collections import OrderedDict

import torch
from torch import Tensor, nn

from evenet.control.config import DotDict, EventInfo
from evenet.network.layers.diffusion.time_conditioned_resnet import TimeConditionedResNet
from evenet.dataset.types import InputType, Source, DistributionInfo

class EventGenerationDecoder(nn.Module):

  def __init__(self, options: DotDict, event_info: EventInfo):
    super(EventGenerationDecoder, self).__init__()
    
    self.input_features        = event_info.input_features
    self.input_types           = event_info.input_types
    self.input_dim             = 0

    for name, source in self.input_features.items():
      if event_info.input_types[name] == InputType.Sequential:
        self.input_dim += 1 # sequential variable having one global num_vector_mean
      if event_info.input_types[name] == InputType.Global:
        self.input_dim += len(source)

    self.projection_dim = options.Network.hidden_dim
    self.resnet = TimeConditionedResNet(options, options.Network.diff_resnet_nlayer, self.input_dim, self.projection_dim, 2* self.projection_dim)

  def forward(self, sources: Tuple[Tensor, Tensor], source_time: Tensor) -> Tuple[Tensor, Tensor]:
    """
      sources: Input[data, mask]
      source_time: Time
      label: condition
    """

    combined_x, mask = sources
    output  = self.resnet(combined_x, source_time)
    return output, mask
