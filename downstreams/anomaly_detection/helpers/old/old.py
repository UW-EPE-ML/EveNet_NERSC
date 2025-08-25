



def select_top_events_fold(true_masses, scores, score_cutoff, plot_bins_left, plot_bins_right, plot_bins_SR):
    
    """
    true_masses: unpreprocessed masses
    """
    # get the events that pass the score threshold
    pass_scores = scores >= score_cutoff

    # correct for diff efficiency in the SB
    SBL_counts_passed, _ = np.histogram(true_masses[pass_scores], bins = plot_bins_left)
    SBL_counts_all, _ = np.histogram(true_masses, bins = plot_bins_left)

    SBH_counts_passed, _ = np.histogram(true_masses[pass_scores], bins = plot_bins_right)
    SBH_counts_all, _ = np.histogram(true_masses, bins = plot_bins_right)

    SR_counts_passed, _ = np.histogram(true_masses[pass_scores], bins = plot_bins_SR)
    SR_counts_all, _ = np.histogram(true_masses, bins = plot_bins_SR)
    
    return true_masses[pass_scores], SBL_counts_passed/SBL_counts_all, SBH_counts_passed/SBH_counts_all, SR_counts_passed/SR_counts_all

   






def calc_significance(masses, fit_function, plot_bins_SR, plot_centers_SR, SR_left, SR_right, popt, pcov = None, ONE_SIDED = False, TWO_SIDED = False):

    num_B_expected_in_SR = sum(fit_function(plot_centers_SR, *popt))
    num_total_in_SR = len(masses[(masses >= SR_left) & (masses <= SR_right)])
    
    num_S_expected_in_SR = num_total_in_SR - num_B_expected_in_SR

    # ONE-SIDED CONSTRAINT
    if ONE_SIDED:

        # Get hist of just B 
        B_function = fit_function(plot_centers_SR, *popt)

        # Get hist of SR data (S+B)
        S_plus_B_function, _ = np.histogram(masses, bins = plot_bins_SR, density = False)

        # If S+B is less than B, then raise the S+B to the B level
        S_plus_B_function = np.where(S_plus_B_function < B_function, B_function, S_plus_B_function)

        # Get the expected number of S events in SR
        num_S_expected_in_SR = sum(S_plus_B_function) - num_B_expected_in_SR


    if TWO_SIDED:

        if pcov is None:
            raise ValueError("Need to provide covariance matrix for two-sided significance calculation")

        # Get the chi2 between the fit and the data in the SR
        B_function = fit_function(plot_centers_SR, *popt) 

        # Get the fit errors using resampling trick
        n = 1000
        temp_params = np.random.multivariate_normal(popt, pcov, n)
        y = np.array([fit_function(plot_centers_SR, *p) for p in temp_params])
        B_error = np.std(y, axis = 0)

        S_plus_B_function, _ = np.histogram(masses, bins = plot_bins_SR, density = False)
        B_error = np.sqrt(B_error**2 + np.sqrt(S_plus_B_function + 1)**2)

        # Get the chi2 between the data and the fit
        chi2 = np.sum((S_plus_B_function - B_function)**2 / B_error**2)
        N_DOF = len(S_plus_B_function) - len(popt)
        chi2_ndof = chi2/N_DOF

        # p_value = 1 - stats.chi2.cdf(chi2, N_DOF)
        log_p_value = stats.chi2.logsf(chi2, N_DOF)
        significance =  np.sqrt(2) * erfcinv(1 * np.exp(log_p_value))
        # significance = stats.norm.ppf(1 - np.exp(log_p_value) / 2)
        # approx_significance = (chi2 - N_DOF) /  np.sqrt(2*N_DOF)

        # print("chi2/ndof:", chi2_ndof, "p_value:", np.exp(log_p_value), "significance:", significance)

        return num_S_expected_in_SR, num_B_expected_in_SR, significance



    return num_S_expected_in_SR, num_B_expected_in_SR



def plot_histograms_with_fits(fpr_thresholds, data_dict_by_fold, scores_dict_by_fold, score_cutoffs_by_fold, mass_scaler, fit_type, title, SB_left, SR_left, SR_right, SB_right, n_folds= 5, take_score_avg=True):
    
    if fit_type == "cubic": fit_function = bkg_fit_cubic
    elif fit_type == "quintic": fit_function = bkg_fit_quintic
    elif fit_type == "septic": fit_function = bkg_fit_septic
    elif fit_type == "ratio": fit_function = bkg_fit_ratio

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right)


    plt.figure(figsize = (12, 9))
    for t, threshold in enumerate(fpr_thresholds):

        # corrections to SR / SB efficiencies
        filtered_masses = []

        for i_fold in range(n_folds):
            loc_true_masses = mass_scaler.inverse_transform(np.array(data_dict_by_fold[i_fold][:,-1]).reshape(-1,1))
            if take_score_avg:
                loc_scores = np.mean(scores_dict_by_fold[i_fold], axis = 1)
            else:
                loc_scores = scores_dict_by_fold[i_fold]
            loc_filtered_masses, loc_SBL_eff, loc_SBH_eff, loc_SR_eff = select_top_events_fold(loc_true_masses, loc_scores, score_cutoffs_by_fold[i_fold][threshold],plot_bins_left, plot_bins_right, plot_bins_SR)
            filtered_masses.append(loc_filtered_masses)

        filtered_masses = np.concatenate(filtered_masses)

        # get the fit function to SB background
        popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(filtered_masses, fit_type, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_all)
        #print("chi2/dof:", chi2/n_dof)
        # plot the fit function
        plt.plot(plot_centers_all, fit_function(plot_centers_all, *popt), lw = 2, linestyle = "dashed", color = f"C{t}")    

        # calculate significance of bump
        num_S_expected_in_SR, num_B_expected_in_SR = calc_significance(filtered_masses, fit_function, plot_bins_SR, SR_left, SR_right, popt)

        y_err = get_errors_bkg_fit_ratio(popt, pcov, plot_centers_SR, fit_type)
        B_error = np.sqrt(np.sum(y_err**2))
        
        S_over_B = num_S_expected_in_SR/num_B_expected_in_SR
        significance = num_S_expected_in_SR/np.sqrt(num_B_expected_in_SR+B_error**2)

        label_string = str(round(100*threshold, 2))+"% FPR: $S/B$: "+str(round(S_over_B,4))+", $S/\sqrt{B}$: "+str(round(significance,4))

        plt.hist(filtered_masses, bins = plot_bins_all, lw = 3, histtype = "step", color = f"C{t}",label = label_string)
        plt.scatter(plot_centers_SB, y_vals, color = f"C{t}")


    plt.legend(loc = (1, 0), fontsize = 24)


    plt.axvline(SR_left, color= "k", lw = 3, zorder = -10)
    plt.axvline(SR_right, color= "k", lw = 3, zorder = -10)

    plt.xlabel("$M_{\mu\mu}$ [GeV]", fontsize = 24)
    plt.ylabel("Counts", fontsize = 24)

    plt.title(title, fontsize = 24)
    

    
    """
def sample_from_flow(model_path, device, context, num_features):
    
    print(f"Loading in the best flow model ...")
    flow_best = torch.load(model_path)
    flow_best.to(device)

    # freeze the trained model
    for param in flow_best.parameters():
        param.requires_grad = False
    flow_best.eval()

    context_masses = torch.tensor(context.reshape(-1,1)).float().to(device)
    SB_samples = flow_best.sample(1, context=context_masses).detach().cpu().numpy()
    SB_samples = SB_samples.reshape(SB_samples.shape[0], num_features)

    SB_samples = np.hstack((SB_samples, np.reshape(context, (-1, 1))))
    
    return SB_samples


def convert_to_latent_space(samples_to_convert, flow_training_dir, training_config_string, device):


    checkpoint_path = os.path.join(flow_training_dir, f"flow_best_model.pt")
    flow_best = torch.load(checkpoint_path)
    flow_best.to(device)

    # freeze the trained model
    for param in flow_best.parameters():
        param.requires_grad = False
    flow_best.eval()

    context_masses = torch.tensor(samples_to_convert[:,-1].reshape(-1,1)).float().to(device)
    outputs_normal_target, _ = flow_best._transform(torch.tensor(samples_to_convert[:,:-1]).float().to(device), context=context_masses)
    
    # note that the mass is saved out as well. Necessary for evaluating the test set
    return np.hstack([outputs_normal_target.detach().cpu().numpy(), samples_to_convert[:,-1].reshape(-1,1)])

"""






