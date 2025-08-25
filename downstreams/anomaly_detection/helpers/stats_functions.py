import numpy as np
import matplotlib.pyplot as plt
import numdifftools
import pickle

from scipy.special import erfcinv, loggamma, erf
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks
from scipy import stats

from helpers.physics_functions import get_bins


"""
CURVE FITTING
"""

def parametric_fit(x, *theta):

    degree = len(theta) - 1
    y = np.zeros_like(x)
    for i in range(degree + 1):
        y += theta[i] * (x)**i

    return y


n_params_DCB = 7
def DCB(x, mu, sigma, N, alpha_L, n_L, alpha_R, n_R):

    n_L = np.abs(n_L)
    n_R = np.abs(n_R)

    A_L = ((n_L/np.abs(alpha_L))**n_L)*np.exp(-np.abs(alpha_L)**2 / 2.0)
    A_R = ((n_R/np.abs(alpha_R))**n_R)*np.exp(-np.abs(alpha_R)**2 / 2.0)

    B_L = (n_L / np.abs(alpha_L)) - np.abs(alpha_L)
    B_R = (n_R / np.abs(alpha_R)) - np.abs(alpha_R)

    # normalization factor
    norm = sigma*((A_L*(alpha_L + B_L)**(1.0 - n_L) / (-1.0 + n_L)) +  (A_R*(alpha_R + B_R)**(1.0 - n_R) / (-1.0 + n_R)) +  np.sqrt(np.pi/2)*(erf(alpha_L/np.sqrt(2))+erf(alpha_R/np.sqrt(2))))

    y = np.zeros_like(x)
    z = (x - mu) / sigma
    # left tail
    y += np.where(z < -alpha_L, A_L*(B_L - z)**(-n_L), 0)
    # right tail
    y += np.where(z > alpha_R, A_R*(B_R + z)**(-n_R), 0)  
    # gaussian core
    y += np.where((z >= -alpha_L) & (z <= alpha_R), np.exp(-0.5*z**2), 0)  
    return N*y/norm

n_params_gaussian = 3
def gaussian(x, mu, sigma, N):
    
    # normalization factor
    norm = 1.0 / np.sqrt(2.0*np.pi*sigma**2)

    y = np.zeros_like(x)
    y += np.exp(-0.5*(x - mu)**2/sigma**2) 
    return N*y/norm
    


def integral(lower, upper, bin_width, *theta):

    degree = len(theta) - 1
    integral = 0
    for i in range(degree + 1):
        integral += theta[i] / (i + 1) * ((upper)**(i + 1) - (lower)**(i + 1))

    return integral / bin_width


def calculate_chi2(y_fit, y_true, sigma):
    return np.sum((y_fit - y_true)**2 / sigma**2)

    
def curve_fit_m_inv(masses, fit_degree, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, N_peaks=0, 
                    weights=None, p0_bkg=None, distance_peaks=3, verbose=False):

    """
    "
    "
    Function to fit background + DCB to mass

    Returns: popt, pcov, chi2, y_vals, n_dof_fit, fit_function (if N_peaks = 0)
    "
    "
    """
    
    if weights is None:
        weights = np.ones_like(masses)

    """
    HISTOGRAM MASSES
    """
   
    # get left SB data
    loc_bkg_left = masses[masses < SR_left]
    weights_left = weights[masses < SR_left]
    plot_centers_left = 0.5*(plot_bins_left[1:]+plot_bins_left[:-1])
    y_vals_left, y_counts_left, bins_left, bin_weights_left, bin_err_left = build_histogram(loc_bkg_left, weights_left, plot_bins_left)

    # get right SB data
    loc_bkg_right = masses[masses > SR_right]
    weights_right = weights[masses > SR_right]
    plot_centers_right = 0.5*(plot_bins_right[1:]+plot_bins_right[:-1])
    y_vals_right, y_counts_right, bins_right, bin_weights_right, bin_err_right = build_histogram(loc_bkg_right, weights_right, plot_bins_right)

    # concatenate the SB data
    y_vals = np.concatenate((y_vals_left, y_vals_right))
    errs = np.concatenate((bin_err_left, bin_err_right))
    y_err = np.sqrt(errs**2 + 1)


    if N_peaks == 0: #  NO PEAKS -- JUST FIT POLYNOMIAL
        average_bin_count = np.mean(y_vals)

        # initialize p0
        p0 = [average_bin_count] + [0 for i in range(fit_degree)]
        n_params_fit = fit_degree + 1
        fit_function = parametric_fit
        
        # set bounds for the curve_fit optimization (not really necessary for polynomial)
        lower_bounds = [-np.inf for x in range(n_params_fit)]
        upper_bounds = [np.inf for x in range(n_params_fit)]
        
        # fit the SB data with regular curvefit
        popt, pcov = curve_fit(fit_function, plot_centers_SB, y_vals, p0, sigma = y_err, maxfev=20000, bounds = (lower_bounds, upper_bounds))


    elif N_peaks > 0: # N PEAKS -- FIT POLYNOMIAL + DCB

        if p0_bkg is None:
            print("Must supply background guess.")
            return

        """ FIRST DO A CURVE_FIT TO GAUSSIAN TO GET P0 FOR DCB """
         
        # define the fit function in-place with N_peaks
        def fit_function_gaussian(x, *theta):
            y = np.zeros_like(x)
            for n_peak in range(N_peaks):
                # sum DCBs
                y += gaussian(x, theta[n_peak*n_params_gaussian], theta[n_peak*n_params_gaussian+1], theta[n_peak*n_params_gaussian+2], )
            # polynomial 
            y += parametric_fit(x, *theta[N_peaks*n_params_gaussian:])
            return y

        # initialize p0
        # find peaks by looking for local maxima, i.e. look for bins that are greater than
                # their distance_peaks neighbors. The local maxima with the N_peaks highest counts (y values)
                # are used for the p0 initialization
        found_peak_indices, found_peak_info = find_peaks(y_vals, distance=distance_peaks, height=0)
        high_to_low_peaks = np.argsort(found_peak_info["peak_heights"])[::-1]
        high_to_low_peak_x = plot_centers_SB[found_peak_indices[high_to_low_peaks]]
        high_to_low_peak_y = found_peak_info["peak_heights"][high_to_low_peaks]

        # p0_DCB contains mu, sigma, N, alpha_L, n_L, alpha_R, n_R
        p0_gauss_init = []
        for n_peak in range(N_peaks):
            p0_gauss_init += [high_to_low_peak_x[n_peak], 0.5*distance_peaks*(plot_centers_SB[1] - plot_centers_SB[0]), high_to_low_peak_y[n_peak]]
        p0_gauss_init += p0_bkg

        # fit the SB data with regular curvefit to get an initial guess for popt
        p0_gaussian_final, _ = curve_fit(fit_function_gaussian, plot_centers_SB, y_vals, p0_gauss_init, sigma = y_err, maxfev=20000)
        

        """ THEN DO A PROPER MINIMIZATION WITH DCB """
        
        # define the fit function in-place with N_peaks
        def fit_function_DCB(x, *theta):
            y = np.zeros_like(x)
            for n_peak in range(N_peaks):
                # sum DCBs
                y += DCB(x, theta[n_peak*n_params_DCB], theta[n_peak*n_params_DCB+1], theta[n_peak*n_params_DCB+2], 
                         theta[n_peak*n_params_DCB+3], theta[n_peak*n_params_DCB+4], theta[n_peak*n_params_DCB+5], theta[n_peak*n_params_DCB+6])
            # polynomial 
            y += parametric_fit(x, *theta[N_peaks*n_params_DCB:])
            return y

        # define the likelihood in-place
        def likelihood(theta):
             # Log poisson likelihood for the SB bins
            log_likelihood = 0
            fit_vals_left = fit_function_DCB(plot_centers_left, *theta)
            fit_vals_right = fit_function_DCB(plot_centers_right, *theta)
            
            log_likelihood += binned_likelihood(y_vals_left, y_counts_left, bin_weights_left, fit_vals_left)
            log_likelihood += binned_likelihood(y_vals_right, y_counts_right, bin_weights_right, fit_vals_right)
              
            return -2 * log_likelihood

        p0_DCB_init = []
        for n_peak in range(N_peaks):
            # pull mu, sigma, N from the gaussian fit
            p0_DCB_init += list(p0_gaussian_final[n_peak*n_params_gaussian:(n_peak+1)*n_params_gaussian])
            # alpha_L, n_L, alpha_R, n_R
            p0_DCB_init += [2, 2, 2, 2] 
        p0_DCB_init += list(p0_gaussian_final[n_params_gaussian*N_peaks:])
        n_params_fit = len(p0_DCB_init)

        # set bounds for the optimizations (the bound n > 1 is necessary. The others are just to help optimization)
        lower_bounds = [np.min(plot_bins_left), 0, 0, 0, 1, 0, 1]*N_peaks + [-np.inf for x in range(fit_degree+1)]
        upper_bounds = [np.max(plot_bins_right), np.max(plot_bins_right)-np.min(plot_bins_right), np.inf, np.inf, np.inf, np.inf, np.inf]*N_peaks + [np.inf for x in range(fit_degree+1)]
        bounds =  [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
        # then the SB data with scipy minimize
        fit = minimize(likelihood, x0 = p0_DCB_init, method = 'Nelder-Mead', options = {'maxiter': 15090, "disp": verbose}, bounds = bounds)
        popt = fit.x
        pcov = None

        fit_function = fit_function_DCB
    
    # get chi2 in the SB
    chi2 = calculate_chi2(fit_function(plot_centers_SB, *popt), y_vals, y_err)

    if N_peaks == 0:
        return popt, pcov, chi2, y_vals, len(y_vals) - n_params_fit
    else:
        return popt, pcov, chi2, y_vals, len(y_vals) - n_params_fit, fit_function

def check_bkg_for_peaks(masses_to_fit, bkg_fit_degree, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, alpha=0.05, verbose=False):

    # fit 0 peaks (only 1 optimization)
    print("Testing bkg fit with 0 peaks...")
    popt_bkg, _, chi2_bkg, _,  n_dof_bkg = curve_fit_m_inv(masses_to_fit, bkg_fit_degree, SR_left, SR_right, plot_bins_left, 
                                                          plot_bins_right, plot_centers_SB)
    
    # fit 1 peak (2 optimizations)
    print("Testing bkg fit with 1 peak...")
    popt_reduced, _, chi2_reduced, _, n_dof_reduced, fit_function_reduced = curve_fit_m_inv(masses_to_fit, bkg_fit_degree, SR_left, SR_right, plot_bins_left, 
                                                          plot_bins_right, plot_centers_SB, N_peaks=1, p0_bkg=list(popt_bkg), verbose=verbose)
    _, p_value_10 = calculate_F_statistic_p_value(chi2_reduced, chi2_bkg, n_dof_reduced, n_dof_bkg)
    
    # fit 2 peak (2 optimizations)
    print("Testing bkg fit with 2 peaks...")
    popt_full, _, chi2_full, _, n_dof_full, fit_function_full = curve_fit_m_inv(masses_to_fit, bkg_fit_degree, SR_left, SR_right, plot_bins_left, 
                                                    plot_bins_right, plot_centers_SB, N_peaks=2, p0_bkg=list(popt_bkg), verbose=verbose)
    _, p_value_21 = calculate_F_statistic_p_value(chi2_full, chi2_reduced, n_dof_full, n_dof_reduced)

    summary_dict = {
        0:{"function":parametric_fit, "popt":popt_bkg, "chi2":chi2_bkg}, 
        1:{"function":fit_function_reduced, "popt":popt_reduced, "chi2":chi2_reduced},
        2:{"function":fit_function_full, "popt":popt_full, "chi2":chi2_full},
                     }
    if verbose: print(f"pval 1-to-0: {p_value_10}; pval 2-to-1: {p_value_21}.")
    if p_value_10 <= alpha: # 1-peak model is better than 0 peaks
        if p_value_21 <= alpha:
            if verbose: print(f"Best model: 2 peaks")
            summary_dict["best"] = {"function":fit_function_full, "popt":popt_full,"n_peaks":2}
        else:
            if verbose: print(f"Best model: 1 peak")
            summary_dict["best"] = {"function":fit_function_reduced, "popt":popt_reduced,"n_peaks":1}
    else:
        if verbose: print(f"Best model: 0 peaks")
        summary_dict["best"] = {"function":parametric_fit, "popt":popt_bkg,"n_peaks":0}
    print()
    print()

    return p_value_10, p_value_21, summary_dict




"""
STATS
"""


def ReLU(x):
    return np.maximum(0, x)

def calculate_F_statistic_p_value(SSE_full, SSE_reduced, n_dof_full, n_dof_reduced):
    numerator = (SSE_reduced - SSE_full) / (n_dof_reduced - n_dof_full)
    denominator = SSE_full / n_dof_full
    Fstar = numerator / denominator
    p_value = 1 - stats.f.cdf(Fstar, n_dof_reduced - n_dof_full, n_dof_full)
    return Fstar, p_value 


def binned_likelihood(yvals, ycounts, weights, fit_vals):

    log_likelihood = 0


    for i in range(len(yvals)):

        expval_weights = np.mean(weights[i])
        expval_weights2 = np.mean(weights[i]**2)
        len_weights = len(weights[i])

        if len_weights == 0 or (np.abs(expval_weights - 1) < 1e-3 and np.abs(expval_weights2 - 1) < 1e-3):
            log_likelihood += stats.poisson.logpmf(yvals[i], fit_vals[i])
        
        else:


            scale_factor = expval_weights2 / expval_weights
            lambda_prime = fit_vals[i] / scale_factor

            n_prime = yvals[i] / scale_factor

            if True:
                log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)

    return log_likelihood

def build_histogram(data, weights, bins):

    y_vals, _bins = np.histogram(data, bins = bins, density = False, weights = weights)
    y_counts, _ = np.histogram(data, bins = bins, density = False)

    digits = np.digitize(data, _bins)
    bin_weights = [weights[digits==i] for i in range(1, len(_bins))]
    bin_err = np.asarray([np.linalg.norm(weights[digits==i]) for i in range(1, len(_bins))])

    return y_vals, y_counts, bins, bin_weights, bin_err


def likelihood(data, s, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR = num_bins)
    plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])


    if weights is None:
        weights = np.ones_like(data)

    # get left SB data
    loc_bkg_left = data[data < SR_left]
    weights_left = weights[data < SR_left]
    y_vals_left, y_counts_left, _bins, left_weights, left_err = build_histogram(loc_bkg_left, weights_left, plot_bins_left)


    # get right SB data
    loc_bkg_right = data[data > SR_right]
    weights_right = weights[data > SR_right]
    y_vals_right, y_counts_right, _bins, right_weights, right_err = build_histogram(loc_bkg_right, weights_right, plot_bins_right)

    # Log poisson likelihood for the SB bins
    log_likelihood = 0
    fit_vals_left = parametric_fit(plot_centers_left, *theta)
    fit_vals_right = parametric_fit(plot_centers_right, *theta)
    
    log_likelihood += binned_likelihood(y_vals_left, y_counts_left, left_weights, fit_vals_left)
    log_likelihood += binned_likelihood(y_vals_right, y_counts_right, right_weights, fit_vals_right)
      

    # get SR data
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    loc_data = data[np.logical_and(data > SR_left, data < SR_right)]
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    err = np.sqrt(np.sum(loc_weights**2))
    # TODO: check if this is correct (tihsu)
    num_bkg = np.clip(integral(SR_left, SR_right, bin_width, *theta), 1e-10, np.inf) # Ensure positive background. tihsu: TODO check correct
    s_prime = s * (s > 0) # Ensure positive signal. If negative, this will cancel out in the likelihood ratio

    expval_weights = np.mean(loc_weights)
    expval_weights2 = np.mean(loc_weights**2)

    # If weights are trivial
    if len(loc_weights) == 0 or (np.abs(expval_weights) < 1e-3 and np.abs(expval_weights2) < 1e-3):

        log_likelihood += stats.poisson.logpmf(num_SR, num_bkg + s_prime)

    else:
        scale_factor = expval_weights2 / expval_weights
        lambda_prime = (num_bkg + s) / scale_factor
        n_prime = num_SR / scale_factor
        log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)
    return -2 * log_likelihood
    
def cheat_likelihood(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR = num_bins)
    plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])
    

    if weights is None:
            weights = np.ones_like(data)

    # get left SB data
    loc_bkg_left = data[data < SR_left]
    weights_left = weights[data < SR_left]
    y_vals_left, y_counts_left, _bins, left_weights, left_err = build_histogram(loc_bkg_left, weights_left, plot_bins_left)

    # get right SB data
    loc_bkg_right = data[data > SR_right]
    weights_right = weights[data > SR_right]
    y_vals_right, y_counts_right, _bins, right_weights, right_err = build_histogram(loc_bkg_right, weights_right, plot_bins_right)
    
    # Log poisson likelihood for the SB bins
    log_likelihood = 0
    fit_vals_left = parametric_fit(plot_centers_left, *theta)
    fit_vals_right = parametric_fit(plot_centers_right, *theta)
   
    log_likelihood += binned_likelihood(y_vals_left, y_counts_left, left_weights, fit_vals_left)
    log_likelihood += binned_likelihood(y_vals_right, y_counts_right, right_weights, fit_vals_right)

    return -2 * log_likelihood
    
def null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta):
    return likelihood(data, 0, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta)


def calculate_test_statistic(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights = None, degree = 5, starting_guess = None, verbose_plot = False, return_popt = False):

    # We want to determine the profiled log likelihood ratio: -2 * [L(s, theta_hat_hat) - L(s_hat, theta_hat)]
    # for s = 0

    # Set up 
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins)
    if weights is None:
        weights = np.ones_like(data)

    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    if starting_guess is None:
        average_bin_count = len(data) / len(plot_centers_all)
        starting_guess = [average_bin_count, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    # Fit the s = 0 hypothesis
    lambda_null = lambda theta: null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta)
    fit = minimize(lambda_null , x0 = starting_guess, method = 'Nelder-Mead', options = {'maxiter': 15090, "disp": verbose_plot})
    theta_hat_hat = fit.x
    null_fit_likelihood = null_hypothesis(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta_hat_hat)


        # Fit the s = float hypothesis
    lambda_cheat = lambda theta: cheat_likelihood(data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta)
    fit = minimize(lambda_cheat , x0 = theta_hat_hat, method = 'Nelder-Mead', options = {'maxiter': 15090, "disp": verbose_plot})
    theta_hat = fit.x
    integrated_background = integral(SR_left, SR_right, bin_width, *theta_hat)
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    integrated_signal = num_SR - integrated_background
    best_fit_likelihood = likelihood(data, integrated_signal, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta_hat)


    # Calculate the test statistic
    test_statistic = (null_fit_likelihood - best_fit_likelihood)
    if integrated_signal < 0:
        test_statistic = 0
    if test_statistic < 0:
        test_statistic = 0

    if verbose_plot:
        print('Best fit:', best_fit_likelihood)
        print('Null fit:', null_fit_likelihood)
        print('Test statistic:', null_fit_likelihood - best_fit_likelihood)
        print('Integrated signal:', integrated_signal)
        print('Integrated background:', integrated_background)


    if verbose_plot:
        plt.hist(data, bins = plot_bins, histtype = 'step', color = 'black', label = 'Data')
        plt.plot(plot_centers, parametric_fit(plot_centers, *theta_hat), label = 'Fit')
        plt.plot(plot_centers, parametric_fit(plot_centers, *theta_hat_hat), label = 'Null')
        plt.legend()


    if return_popt:
        return integrated_signal, integrated_background, test_statistic, theta_hat

    return integrated_signal, integrated_background, test_statistic

    