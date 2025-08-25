import numpy as np
from scipy.stats import ks_2samp


def get_kl_dist(samp0, samp1):
    
    divs = []
    for i in range(samp0.shape[1]):
        divs.append(ks_2samp(samp0[:,i], samp1[:,i])[0])
    return divs

def get_median_percentiles(x_array):
    
    x_median = np.median(x_array, axis = 1)
    x_lower = np.percentile(x_array, 16, axis = 1)
    x_upper = np.percentile(x_array, 84, axis = 1)

    return x_median, x_lower, x_upper

"""
LaCATHODE option
"""

from helpers.density_estimator import DensityEstimator
import torch

def convert_to_latent_space_true_cathode(samples_to_convert, num_inputs, flow_training_dir, config_file, device):
    
    val_losses = np.load(os.path.join(flow_training_dir, "flow_val_losses.npy"))

    # get epoch of best val loss
    best_epoch = np.argmin(val_losses) - 1
    model_path = f"{flow_training_dir}/flow_epoch_{best_epoch}.par"

    eval_model = DensityEstimator(config_file, num_inputs, eval_mode=True, load_path=model_path, device=device, verbose=False,bound=False)
    
    context_masses = torch.tensor(samples_to_convert[:,-1].reshape(-1,1)).float().to(device)
    
    outputs_normal_target = eval_model.model.forward(torch.tensor(samples_to_convert[:,:-1]).float().to(device), context_masses, mode='direct')[0]
    return np.hstack([outputs_normal_target.detach().cpu().numpy(), samples_to_convert[:,-1].reshape(-1,1)])
    


