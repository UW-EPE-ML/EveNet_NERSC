import numpy as np

from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform

from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal

from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.lu import LULinear

from helpers.dense import dense_net


def assemble_masked_AR_transforms(num_features, num_layers, num_hidden_features, num_blocks, num_bins=10, spline_type="PRQ", perm_type="Reverse", tail_bound=3, tails="linear", dropout=0, use_batch_norm=True):
    
    transforms = []
    
    for _ in range(num_layers):
        
        if spline_type == "PRQ":
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features, hidden_features=num_hidden_features, num_blocks=num_blocks, tail_bound=tail_bound, context_features=1, tails=tails, num_bins=num_bins, dropout_probability=dropout, use_batch_norm=use_batch_norm))   
            
        elif spline_type == "MAF":
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features, hidden_features=num_hidden_features, num_blocks=num_blocks, context_features=1, dropout_probability=dropout, use_batch_norm=use_batch_norm))
            
        if perm_type == "Reverse":
            transforms.append(ReversePermutation(features=num_features)) 
        else:
            transforms.append(LULinear(features=num_features)) 
            
    return transforms

"""

def assemble_coupling_transforms(num_features, num_layers, num_hidden_features, num_hidden_layers, num_bins = 10, spline_type = "PiecewiseRationalQuadratic", perm_type = "Reverse", tail_bound = 3, tails = "linear"):
    
    # first make the mask
    n_mask = int(np.ceil(num_features / 2))
    mx = [1] * n_mask + [0] * int(num_features - n_mask)

    # then make the maker
    # this has to be an nn.module that takes as first arg the input dim and second the output dim
    def maker(input_dim, output_dim):
        return dense_net(input_dim, output_dim, layers=[num_hidden_features] * num_hidden_layers, context_features=1)

    transforms = []

    for _ in range(num_layers):

        if spline_type == "PiecewiseRationalQuadratic":
            transforms.append(PiecewiseRationalQuadraticCouplingTransform(mx, maker, tail_bound = tail_bound, tails = tails, num_bins = num_bins))      
        
        if perm_type == "Reverse":
            transforms.append(ReversePermutation(features = num_features)) 
        else:
            transforms.append(LULinear(features = num_features)) 
            
    return transforms
"""


def make_masked_AR_flow(num_features, num_layers, num_hidden_features, num_blocks, spline_type):

    # Define a flow architecture
    transforms = assemble_masked_AR_transforms(num_features, num_layers, num_hidden_features, num_blocks, spline_type=spline_type)
    base_dist = StandardNormal(shape=[num_features])
    
    flow = Flow(CompositeTransform(transforms), base_dist)
    
    return flow

"""
def make_coupling_flow(num_features, num_layers, num_hidden_features, num_hidden_layers):

    # Define a flow architecture
    transforms = assemble_coupling_transforms(num_features, num_layers, num_hidden_features, num_hidden_layers)
    base_dist = StandardNormal(shape=[num_features])
    
    flow = Flow(CompositeTransform(transforms), base_dist)
    
    return flow
"""