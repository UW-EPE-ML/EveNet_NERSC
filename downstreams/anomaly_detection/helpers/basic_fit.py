
import vector
vector.register_awkward()

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotting import *
from physics_functions import get_bins
from stats_functions import parametric_fit, curve_fit_m_inv, calculate_test_statistic
from data_transforms import logit_transform
import pyarrow.parquet as pq
import numpy as np
from scipy.interpolate import interp1d

def plot_mass_distribution(
    inv_mass,
    SR_left,
    SR_right,
    SB_left,
    SB_right,
    bkg_fit_degree = 5,
    num_bins_SR = 20,
    fit_stat=False,
    save_name = None
):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR= num_bins_SR)

    x = np.linspace(SB_left, SB_right, 100) # plot curve fit

    plt.figure(figsize = (10,5))

    if not isinstance(inv_mass, np.ndarray):
        inv_mass = inv_mass.to_numpy()
    # curve fit the data
    popt_0, _, _, _, _ = curve_fit_m_inv(inv_mass, bkg_fit_degree, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)
    # plot the best fit curve
    # cßalculate the test statistic
    plt.plot(x, parametric_fit(x, *popt_0), lw = 3, linestyle = "dashed")
    if fit_stat:
        S, B, q0 = calculate_test_statistic(inv_mass, SR_left, SR_right, SB_left, SB_right, num_bins_SR, degree = bkg_fit_degree, starting_guess = popt_0)
    else:
        q0 = 0
    # plot all data
    plt.hist(inv_mass, bins = plot_bins_all, lw = 2, histtype = "step", density = False, label = f"sig: {round(np.sqrt(q0),3)}")

    plt.axvline(SR_left)
    plt.axvline(SR_right)

    plt.xlabel("Dijet M [GeV]")
    plt.ylabel("Counts")
    plt.legend(loc = "upper right")

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    return {
        "plot_bins_all": plot_bins_all,
        "plot_bins_SR": plot_bins_SR,
        "plot_bins_left": plot_bins_left,
        "plot_bins_right": plot_bins_right,
        "plot_centers_all": plot_centers_all,
        "plot_centers_SR": plot_centers_SR,
        "plot_centers_SB": plot_centers_SB,
        "popt_0": popt_0,
    }


