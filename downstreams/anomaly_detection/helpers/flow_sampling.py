import zfit
from zfit import z
import numpy as np
import torch
import tensorflow as tf


def get_flow_samples(model, masses):
    with torch.no_grad():
        feats = model.model.sample(num_samples=masses.shape[0], cond_inputs=torch.tensor(masses.reshape(-1,1)).float()).detach().cpu().numpy()
    return np.hstack([feats, masses.reshape(-1,1)])

def get_mass_samples(SR_left, SR_right, n_SR_samples, popt, fit_function):

    # necessary to convert between torch and numpy
    tf.config.run_functions_eagerly(True)
    
    obs = zfit.Space("mass_inv", limits=(SR_left, SR_right))
    class CustomPDF(zfit.pdf.ZPDF):
        _PARAMS = tuple([f"a{i}" for i in range(len(popt))])
        def _unnormalized_pdf(self, x):
            data = z.unstack_x(x)
            y = tf.zeros_like(data)
            y += torch.from_numpy(fit_function(data.numpy(), *popt))
            return y

    extra_args = {f"a{i}":popt[i] for i in range(len(popt))}
    custom_pdf = CustomPDF(obs=obs, **extra_args)
    
    return custom_pdf.sample(n=n_SR_samples)["mass_inv"].numpy()
    
    