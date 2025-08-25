import numpy as np
import matplotlib.pyplot as plt
import numdifftools
import pickle

from scipy.special import erfcinv
from scipy.optimize import curve_fit, minimize
from scipy import stats


muon_mass = 0.1056583755 # GeV

def assemble_m_inv(a_M, a_pt, a_eta, a_phi, b_M, b_pt, b_eta, b_phi):
    # computes system of mother particle
    
    a_E = np.sqrt(a_M**2 + (a_pt*np.cosh(a_eta))**2)
    b_E = np.sqrt(b_M**2 + (b_pt*np.cosh(b_eta))**2)

    a_px = a_pt*np.cos(a_phi)
    b_px = b_pt*np.cos(b_phi)

    a_py = a_pt*np.sin(a_phi)
    b_py = b_pt*np.sin(b_phi)

    a_pz = a_pt*np.sinh(a_eta)
    b_pz = b_pt*np.sinh(b_eta)

    mother_E = a_E + b_E
    mother_px = a_px + b_px
    mother_py = a_py + b_py
    mother_pz = a_pz + b_pz
    M_sq_cands = mother_E**2 - mother_px**2 - mother_py**2 - mother_pz**2
    mother_M = np.sqrt(M_sq_cands)
    mother_pt = np.sqrt(mother_px**2 + mother_py**2)
    mother_eta = np.arcsinh(mother_pz/mother_pt)
    mother_phi = np.arctan2(mother_py, mother_px)

    good_event_indices = (M_sq_cands >= 0) & (mother_pt > 0)
    
    return mother_M, mother_pt, mother_eta, mother_phi, good_event_indices


def calculate_deltaR(phi_0, phi_1, eta_0, eta_1):
    
    delta_phi = np.abs(phi_0 - phi_1)
    # adjust to be in the range(-pi, pi)
    delta_phi = np.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
    delta_R = np.sqrt(delta_phi**2 + (eta_0 - eta_1)**2)
    
    return delta_R

"""
"
"
BINNING
"
"
"""

def get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR, binning="linear"):
    
    if binning == "linear":

        plot_bins_SR = np.linspace(SR_left, SR_right, num_bins_SR)
        plot_centers_SR = 0.5*(plot_bins_SR[1:] + plot_bins_SR[:-1])
        width = plot_bins_SR[1] - plot_bins_SR[0]

        plot_bins_left = np.arange(SR_left, SB_left-width,  -width)[::-1]
        if plot_bins_left[0] < SB_left:
            plot_bins_left = plot_bins_left[1:]
        plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])

        plot_bins_right = np.arange(SR_right, SB_right+width, width)
        if plot_bins_right[-1] > SB_right:
            plot_bins_right = plot_bins_right[:-1]
        plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])
        
    elif binning == "log":
        plot_bins_SR = np.logspace(np.log10(SR_left), np.log10(SR_right), num_bins_SR)
        plot_centers_SR = np.array([np.sqrt(plot_bins_SR[i]*plot_bins_SR[i+1]) for i in range(len(plot_bins_SR)-1)])
        ratio = plot_bins_SR[1]/plot_bins_SR[0]
        
        # SBL
        current_endpoint = plot_bins_SR[0]
        plot_bins_left = [current_endpoint]
        while current_endpoint > SB_left:
            next_endpoint = current_endpoint/ratio
            plot_bins_left.insert(0, next_endpoint)
            current_endpoint = next_endpoint
        if plot_bins_left[0] < SB_left:
            plot_bins_left = plot_bins_left[1:]
        plot_centers_left = np.array([np.sqrt(plot_bins_left[i]*plot_bins_left[i+1]) for i in range(len(plot_bins_left)-1)])
        
        # SBR
        current_endpoint = plot_bins_SR[-1]
        plot_bins_right = [current_endpoint]
        while current_endpoint < SB_right:
            next_endpoint = current_endpoint*ratio
            plot_bins_right.append(next_endpoint)
            current_endpoint = next_endpoint
        if plot_bins_right[-1] > SB_right:
            plot_bins_right = plot_bins_right[:-1]
        plot_centers_right = np.array([np.sqrt(plot_bins_right[i]*plot_bins_right[i+1]) for i in range(len(plot_bins_right)-1)])
    
    plot_centers_all = np.concatenate((plot_centers_left, plot_centers_SR, plot_centers_right))
    plot_centers_SB = np.concatenate((plot_centers_left, plot_centers_right))
    plot_bins_all = np.concatenate([plot_bins_left[:-1], plot_bins_SR, plot_bins_right[1:]])
    
    
    return plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB
             
def get_bins_for_scan(path_to_bin_defs_folder, window_index, scale_bins=False):
    
    """
    If this code is run on preprocessed data, then the bin definitions need to be modified with a scaler
    """
    with open(f"{path_to_bin_defs_folder}/bin_definitions", "rb") as infile:
        bin_definitions = pickle.load(infile)
    window_bin_definitions = bin_definitions[window_index]
    
    plot_bins_SR = window_bin_definitions["SR"]
    plot_bins_left = window_bin_definitions["SBL"]
    plot_bins_right = window_bin_definitions["SBH"]
    
    plot_centers_SR = np.array([np.sqrt(plot_bins_SR[i]*plot_bins_SR[i+1]) for i in range(len(plot_bins_SR)-1)])
    plot_centers_left = np.array([np.sqrt(plot_bins_left[i]*plot_bins_left[i+1]) for i in range(len(plot_bins_left)-1)])
    plot_centers_right = np.array([np.sqrt(plot_bins_right[i]*plot_bins_right[i+1]) for i in range(len(plot_bins_right)-1)])

    plot_centers_all = np.concatenate((plot_centers_left, plot_centers_SR, plot_centers_right))
    plot_centers_SB = np.concatenate((plot_centers_left, plot_centers_right))
    plot_bins_all = np.concatenate([plot_bins_left[:-1], plot_bins_SR, plot_bins_right[1:]])
    
    if scale_bins:
        with open(f"{path_to_bin_defs_folder}/mass_scaler_window{window_index}", "rb") as infile:
            mass_scaler = pickle.load(infile)
        plot_bins_all = mass_scaler.transform(np.array(plot_bins_all).reshape(-1,1)).reshape(-1,)
        plot_bins_SR = mass_scaler.transform(np.array(plot_bins_SR).reshape(-1,1)).reshape(-1,)
        plot_bins_left = mass_scaler.transform(np.array(plot_bins_left).reshape(-1,1)).reshape(-1,)
        plot_bins_right = mass_scaler.transform(np.array(plot_bins_right).reshape(-1,1)).reshape(-1,)
        plot_centers_all = mass_scaler.transform(np.array(plot_centers_all).reshape(-1,1)).reshape(-1,)
        plot_centers_SR = mass_scaler.transform(np.array(plot_centers_SR).reshape(-1,1)).reshape(-1,)
        plot_centers_SB = mass_scaler.transform(np.array(plot_centers_SB).reshape(-1,1)).reshape(-1,)
            
    return plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB
    