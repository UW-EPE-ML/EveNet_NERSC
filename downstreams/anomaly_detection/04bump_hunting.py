import argparse
import yaml
import os

import sys
import concurrent.futures

from matplotlib.lines import Line2D
import json
# Add the parent directory of the notebook to the path
sys.path.append(os.path.abspath("../.."))

from rich.console import Console
from rich.table import Table
from rich import box
import matplotlib.pyplot as plt

import numpy as np
import vector
vector.register_awkward()
import pandas as pd

from evenet.control.global_config import global_config

from helpers.physics_functions import get_bins
from helpers.stats_functions import curve_fit_m_inv, parametric_fit, calculate_test_statistic
from helpers.plotting import newplot, function_with_band, stamp
from helpers.utils import clean_and_append

from joblib import Parallel, delayed
from scipy.stats import norm

bin_percent = {8: 2.3, 12: 1.5, 16: 1.1}
fit_type = {3: "cubic", 5: "quintic", 7: "septic"}

def inverse_quantile(sigma):
    return 1 - norm.cdf(sigma)


def plot_upsilon_resonances(ax):
    # plot the upsilon resonances
    ax.axvline(9.460, color="black", linestyle="--", alpha=0.2, lw=2)
    ax.text(9.460 * 0.995, 1e4, r"$\Upsilon(1S)$", rotation=90, verticalalignment="center", horizontalalignment="right",
            fontsize=12)
    ax.axvline(10.023, color="black", linestyle="--", alpha=0.2, lw=2)
    ax.text(10.023 * 0.995, 1e4, r"$\Upsilon(2S)$", rotation=90, verticalalignment="center",
            horizontalalignment="right", fontsize=12)
    ax.axvline(10.355, color="black", linestyle="--", alpha=0.2, lw=2)
    ax.text(10.355 * 0.995, 1e4, r"$\Upsilon(3S)$", rotation=90, verticalalignment="center",
            horizontalalignment="right", fontsize=12)
    # ax.axvline(10.580, color="black", linestyle="--", alpha=0.15, lw = 1.5)
    # ax.text(10.580 * 0.995, 1e4, r"$\Upsilon(4S)$", rotation=90, verticalalignment="center", horizontalalignment="right", fontsize=5)


def plot_histograms_with_fits(
        fpr_thresholds,
        data_dict,
        fit_degree,
        title,
        SB_left,
        SR_left,
        SR_right,
        SB_right,
        num_bins_SR,
        n_folds=5,
        take_score_avg=True,
        latex_flag=False,
        ymin=1.0,
        ymax=1e5,
        ncpu=1,
        legend_title=r"$\bf{CMS 2016 Open Data: Dimuons}$",
        fit=False,
        score_name="xgb_prob",
        fit_result = None,
        channel = "OS"
):
    save_data = {}
    save_data["fpr_thresholds"] = fpr_thresholds
    save_data["fit_degree"] = fit_degree
    save_data["num_bins_SR"] = num_bins_SR
    save_data["popts"] = []
    save_data["pcovs"] = []
    save_data["significances"] = []
    save_data["filtered_masses"] = []
    save_data["y_vals"] = []

    # define bins and bin edges for the SB and SR
    # change the bin width with `num_bins_SR`
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(
        SR_left, SR_right, SB_left, SB_right, num_bins_SR=num_bins_SR)

    # Get a list of all possible cuts for the feature

    all_scores = data_dict[score_name]
    all_masses = data_dict["inv_mass"]
    in_SR = (all_masses >= SR_left) & (all_masses <= SR_right)
    in_SBL = (all_masses < SR_left)
    in_SBH = (all_masses > SR_right)

    mass_SBL = all_masses[in_SBL]
    mass_SR = all_masses[in_SR]
    mass_SBH = all_masses[in_SBH]

    feature_SBL = all_scores[in_SBL]
    feature_SR = all_scores[in_SR]
    feature_SBH = all_scores[in_SBH]

    feature_cut_points = np.linspace(np.min(all_scores), np.max(all_scores), 10000)
    # For each cut, calculate the number of signal and background events in the SR
    num_in_SBL = []
    num_in_SR = []
    num_in_SBH = []
    FPR = []
    for cut in feature_cut_points:
        num_in_SBL.append(np.sum(feature_SBL >= cut) / len(feature_SBL))
        num_in_SR.append(np.sum(feature_SR >= cut) / len(feature_SR))
        num_in_SBH.append(np.sum(feature_SBH >= cut) / len(feature_SBH))

        FPR.append((np.sum(feature_SBH >= cut) + np.sum(feature_SBL >= cut)) / (len(feature_SBH) + len(feature_SBL)))

    fig, ax = newplot("full", width=12, height=9, use_tex=latex_flag)

    def process_threshold(t, threshold):
        # Best cut at this threshold
        best_feature_cut = feature_cut_points[np.argmin(np.abs(np.array(FPR) - threshold))]

        mass_SBL_cut = mass_SBL[feature_SBL >= best_feature_cut]
        mass_SR_cut = mass_SR[feature_SR >= best_feature_cut]
        mass_SBH_cut = mass_SBH[feature_SBH >= best_feature_cut]

        filtered_masses = np.concatenate((mass_SBL_cut, mass_SR_cut, mass_SBH_cut))

        # Fit
        popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(
            filtered_masses, fit_degree, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB
        )

        # Test statistic
        if fit:
            S, B, q0 = calculate_test_statistic(
                filtered_masses, SR_left, SR_right, SB_left, SB_right, num_bins_SR,
                degree=fit_degree, starting_guess=popt
            )
        else:
            S, B, q0 = 10, 100, 0

        total_events = len(filtered_masses)

        label_string = f"{round(100 * threshold, 2)}% FPR: {total_events} events,  $Z_0$: {round(np.sqrt(q0), 2)}"

        # Return everything needed
        return {
            "t": t,
            "threshold": threshold,
            "filtered_masses": filtered_masses,
            "popt": popt,
            "pcov": pcov,
            "significance": np.sqrt(q0),
            "y_vals": y_vals,
            "label_string": label_string,
        }

    if fit_result is None:
        results = Parallel(n_jobs=ncpu)(delayed(process_threshold)(t, thr) for t, thr in enumerate(fpr_thresholds))
        indices = np.linspace(0, len(results) - 1, len(results), dtype=int)
    else:
        print("Using provided fit result")
        results = []
        indices = []
        for idx in range(len(fit_result["popts"])):
            result = {
                "t": idx,
                "threshold": fpr_thresholds[idx],
                "filtered_masses": fit_result["filtered_masses"][idx],
                "popt": fit_result["popts"][idx],
                "pcov": fit_result["pcovs"][idx],
                "significance": fit_result["significances"][idx],
                "y_vals": fit_result["y_vals"][idx],
                # "label_string": f"{round(100 * fpr_thresholds[idx], 2)}% FPR: {len(fit_result['filtered_masses'][idx])} events,  $Z_0$: {round(np.sqrt(fit_result['significances'][idx]), 2)}"
                "label_string": f"{round(100 * fpr_thresholds[idx], 2)}%, {fit_result['significances'][idx]:.2f}$\sigma$"

            }
            results.append(result)
            indices.append(idx)


    colorblind_colors = [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#000000",  # black
    ]
    plot_fpr = [1.0, 0.1, 0.01, 0.001]
    for idx, result in enumerate(results):
        t = idx
        color = colorblind_colors[idx % len(colorblind_colors)]

        if fpr_thresholds[t] in plot_fpr:
            plt.plot(plot_centers_all, parametric_fit(plot_centers_all, *result["popt"]), lw=2, linestyle="dashed", color=color)
            function_with_band(ax, parametric_fit, [SB_left, SB_right], result["popt"], result["pcov"], color=color)
            plt.hist(result["filtered_masses"], bins=plot_bins_all, lw=10, histtype="step", color=color, label=result["label_string"], alpha=0.75)
            plt.scatter(plot_centers_SB, result["y_vals"], color=color)

        save_data["popts"].append(result["popt"])
        save_data["pcovs"].append(result["pcov"])
        save_data["significances"].append(result["significance"])
        save_data["filtered_masses"].append(result["filtered_masses"])
        save_data["y_vals"].append(result["y_vals"])


    if channel == "OS":
        line_0 = r"$\mathbf{Opposite\ Sign}$"
    else:
        line_0 = r"$\mathbf{Same\ Sign}$"
    line_1 = f"$\mathrm{{Bin\ width}} = {bin_percent[num_bins_SR]}\%$"
    line_2 = f"$\mathrm{{Fit\ Type:}}$ {fit_type[fit_degree].capitalize()}"
    line_3 = r"$\mathrm{Anti-Isolated}$"
    line_4 = r"$8.7\ \mathrm{fb}^{-1},\ \sqrt{s} = 13\ \mathrm{TeV}$"

    starting_x = 0.050
    starting_y = 0.95
    delta_y = 0.05
    text_alpha = 0.85

    if line_0 is not None:
        ax.text(starting_x, starting_y - (0) * delta_y, line_0, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', alpha=text_alpha, zorder=10)
        ax.text(starting_x, starting_y - (1) * delta_y, line_1, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', alpha=text_alpha, zorder=10)
        ax.text(starting_x, starting_y - (2) * delta_y, line_2, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', alpha=text_alpha, zorder=10)
        ax.text(starting_x, starting_y - (3) * delta_y, line_3, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', alpha=text_alpha, zorder=10)
        ax.text(starting_x, starting_y - (4) * delta_y, line_4, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', alpha=text_alpha, zorder=10)

    # line1 = f"{num_bins_SR - 1} Bins in SR"
    # line2 = f"Fit Type: {fit_degree}-degree Polynomial"
    # # line3 = r"Muon Iso_04 $\geq$ 0.55"
    # # line4 = r"~6% of Original Data"
    #
    # starting_x = 0.05
    # starting_y = 0.8
    # delta_y = 0.04
    # text_alpha = 0.75
    # ax.text(starting_x, starting_y - 0 * delta_y, line1, transform=ax.transAxes, fontsize=14, verticalalignment='top',
    #         alpha=text_alpha)
    # ax.text(starting_x, starting_y - 1 * delta_y, line2, transform=ax.transAxes, fontsize=14, verticalalignment='top',
    #         alpha=text_alpha)
    # ax.text(starting_x, starting_y - 2 * delta_y, line3, transform=ax.transAxes, fontsize=14, verticalalignment='top', alpha = text_alpha)
    # ax.text(starting_x, starting_y - 3 * delta_y, line4, transform=ax.transAxes, fontsize=14, verticalalignment='top', alpha = text_alpha)

    plt.legend(loc=(0.68, 0.72), fontsize=14, title="False Positive Rate", title_fontsize=16)
    legend_title = r"$\bf{2016\,\, CMS\,\, Open\,\, Data\,\, DoubleMuon}$"
    plt.title(legend_title, loc = "right", fontsize = 16)


    plt.axvline(SR_left, color="k", lw=3, zorder=10)
    plt.axvline(SR_right, color="k", lw=3, zorder=10)

    plt.xlabel("$M_{\mu\mu}$ [GeV]", fontsize=18)
    plt.ylabel("Events", fontsize=18)

    plt.yscale("log")
    # plt.ylim(0.5, 1e3)
    plt.ylim(ymin, ymax)
    plt.xlim(SB_left, SB_right)

    # Add more x ticks (major and minor)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.tick_params(axis='y', which='minor', left=True)

    # Do the thing for the pre-isolation data
    filtered_masses = data_dict["inv_mass"]

    # get the fit function to SB background


    # print("chi2/dof:", chi2/n_dof)
    if fit_result is None:

        popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(
            filtered_masses, fit_degree, SR_left, SR_right,
            plot_bins_left,
            plot_bins_right, plot_centers_SB
        )
        S, B, q0 = calculate_test_statistic(filtered_masses, SR_left, SR_right, SB_left, SB_right, num_bins_SR,
                                            degree=fit_degree, starting_guess=popt)
        # plot the fit function
        plt.plot(plot_centers_all, parametric_fit(plot_centers_all, *popt), lw=2, linestyle="dashed", color=f"black")
        function_with_band(ax, parametric_fit, [SB_left, SB_right], popt, pcov, color=f"black")
        # print(S, B, np.sqrt(q0))
        # print(popt)

        total_events = len(filtered_masses)
        label_string = "Pre-cut; " + str(total_events) + " events,  $Z_0$: " + str(round(np.sqrt(q0), 2))
        plt.hist(filtered_masses, bins=plot_bins_all, lw=3, histtype="step", color=f"black", label=label_string, alpha=0.75)
        plt.scatter(plot_centers_SB, y_vals, color=f"black")

        # save_data["popts"].append(popt)
        # save_data["pcovs"].append(pcov)
        # save_data["significances"].append(np.sqrt(q0))
        # save_data["filtered_masses"].append(filtered_masses)
        # save_data["y_vals"].append(y_vals)

        # l-reweighting
        mu = S / (S + B)
        likelihood_ratios = (all_scores) / (1 - all_scores)
        weights = (likelihood_ratios - (1 - mu)) / mu
        weights = np.clip(weights, 1e-9, 1e9)

        popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(
            all_masses, fit_degree, SR_left, SR_right, plot_bins_left,
            plot_bins_right, plot_centers_SB, weights=weights
        )
        s, b, bonus_q0, popt = calculate_test_statistic(all_masses, SR_left, SR_right, SB_left, SB_right, num_bins_SR,
                                                        weights=weights, degree=fit_degree, verbose_plot=False,
                                                        starting_guess=popt, return_popt=True)

    else:
        bonus_q0 = fit_result["bonus_significance"] * fit_result["bonus_significance"]

    plot_upsilon_resonances(ax)

    print(f"✅ Full Likelihood Fit {num_bins_SR}bins {fit_degree}degree sqrt(q0): ", np.sqrt(bonus_q0))
    ax.text(starting_x, starting_y - 5 * delta_y, f"l-reweighting: {np.sqrt(bonus_q0):.2f}", transform=ax.transAxes,
            fontsize=14, verticalalignment='top', alpha=text_alpha)

    # # Vertical Black Lines at boundaries of SR
    # plt.axvline(SR_left, color = "black", linestyle = "--", lw = 2)
    # plt.axvline(SR_right, color = "black", linestyle = "--", lw = 2)

    # plt.title(title, fontsize = 24)
    return save_data, np.sqrt(bonus_q0)

def plot_significane_vs_pfr(
        fpr_thresholds,
        save_data,
        num_bins_SR,
        fit_degree,
        bonus_significance=None,
        ymin = 1e-15,
        save_name = None):
    fig, ax =  newplot("column", width=4, height=4)

    primary_colors = ["red","orange","green", "blue"]
    colors = ["lightcoral", "gold", "lime", "cornflowerblue"]

    min_x = 2e-4
    labels = {"EveNet-pretrain": r"EveNet-pretrian"}
    linestyles = {"EveNet-pretrain": "-"}
    markersize = {"EveNet-pretrain": 5}

    icolor = 0
    p_values = inverse_quantile(save_data["significances"])
    ax.plot(fpr_thresholds,
            p_values,
            color = primary_colors[icolor],
            lw = 3,
            alpha = 0.75,
            marker = "o",
            markersize = 5,
            label = "EveNet",
            ls = "-"
        )

    if bonus_significance is not None:
        p_value = inverse_quantile(bonus_significance)
        plt.axhline(p_value, color = "purple", lw = 3, alpha = 0.75, ms = 3, label = r"$\ell$-Reweighting")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Background-Only $p$-Value")

    plt.yscale("log")
    line_0 = r"\textbf{Opposite Sign} Muons"


    line1 = f"Bin width = {bin_percent[num_bins_SR]}\%"
    line2 = f"Fit Type: {fit_type[fit_degree].capitalize()}"
    line3 = r"Muon Iso_04 $\geq$ 0.55"
    line4 = r"8.7 fb$^{-1}$, $\sqrt{s} = 13$ TeV"

    starting_x = 0.050
    starting_y = 0.25
    delta_y = 0.05
    text_alpha = 0.75

    if line_0 is not None:
        ax.text(starting_x, starting_y - (-1.5) * delta_y, r"$\texttt{HLT\_TrkMu15\_DoubleTrkMu5NoFiltersNoVt}$",
                transform=ax.transAxes, fontsize=5, verticalalignment='top', alpha=text_alpha, zorder=10)
        ax.text(starting_x, starting_y - (-1) * delta_y, line_0, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', alpha=text_alpha, zorder=10)

    legend_title = r"$\bf{2016\,\, CMS\,\, Open\,\, Data\,\, DoubleMuon}$"
    plt.legend(loc = "lower right", ncol = 1, fontsize = 9)
    plt.title(legend_title, loc = "right", fontsize = 10)
    plt.axvline(1e-4, color = "grey", linestyle = "dashed", alpha = 0.5, lw = 1)

    # Plot sigmas
    i = 0
    while inverse_quantile(i) > ymin:
        p_value = inverse_quantile(i)
        plt.axhline(p_value, color = "grey", linestyle = "dashed", alpha = 0.5, lw = 1)

        if i > 0 and inverse_quantile(i+1) > ymin:
            plt.text(3e-4, p_value * 1.5, f"{i}$\sigma$", fontsize = 10, verticalalignment = "center")

        # fill above
        plt.fill_between([min_x, 1], p_value, 0.5, color = "grey", alpha = 0.025)

        i += 1

    plt.xscale("log")
    plt.ylim(ymin, 0.5)
    plt.xlim(min_x, 1)


    if not save_name is None:
        for name in save_name:
            plt.savefig(name)

    return fig, ax

def plot_significance_vs_fpr_all(
    summary_data,
    ymin=1e-15,
    save_name=None,
    bin_percent=None,
    fit_type=None,
):
    fig, ax = newplot("column", width=4, height=4)


    fit_colors = {
        "cubic": "red",
        "quintic": "purple",
        "septic": "blue"
    }
    bin_styles = {
        1.1: "dashed",
        1.5: "solid",
        2.3: "dotted"
    }

    # Styling maps
    # color_map = {
    #     fit_degree: color for fit_degree, color in zip(sorted(fit_type), ["red", "green", "blue", "orange"])
    # }

    # linestyle_map = {
    #     num_bins_SR: ls for num_bins_SR, ls in zip(sorted(bin_percent), ["-", "--", "-.", ":"])
    # }

    # Labels for legend
    def legend_label(num_bins_SR, fit_degree):
        return f"mass width: {bin_percent[num_bins_SR]}%, {fit_type[fit_degree].capitalize()} fit"

    # Plot each curve
    for num_bins_SR in sorted(summary_data):
        for fit_degree in sorted(summary_data[num_bins_SR]):
            data = summary_data[num_bins_SR][fit_degree]
            fpr_thresholds = data["fpr_thresholds"]
            save_data = data["save_data"]

            p_values = inverse_quantile(save_data["significances"])
            ax.plot(
                fpr_thresholds,
                p_values,
                color=fit_colors[fit_type[fit_degree]],
                linestyle=bin_styles[bin_percent[num_bins_SR]],
                lw=2,
                marker="o",
                markersize=4,
                label=legend_label(num_bins_SR, fit_degree),
                alpha=0.85,
            )

            # bonus_significance_data = data.get("bonus_significance", None)
            # # Optional bonus significance
            # if bonus_significance_data is not None:
            #         bonus_p = inverse_quantile(bonus_significance_data)
            #         ax.axhline(
            #             bonus_p,
            #             color=fit_colors[fit_type[fit_degree]],
            #             lw=4,
            #             linestyle=linestyle_map[num_bins_SR],
            #             label= r"$\ell$-Reweighting"
            #         )

    # Axis settings
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Background-Only $p$-Value")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(2e-4, 1)
    plt.ylim(ymin, 0.5)
    plt.axvline(1e-4, color="grey", linestyle="dashed", alpha=0.5, lw=1)

    # Draw horizontal lines for sigma levels
    i = 0
    while inverse_quantile(i) > ymin:
        p_val = inverse_quantile(i)
        ax.axhline(p_val, color="grey", linestyle="dashed", alpha=0.5, lw=1)
        if i > 0 and inverse_quantile(i + 1) > ymin:
            ax.text(3e-4, p_val * 1.5, f"{i}$\\sigma$", fontsize=10, verticalalignment="center")
        plt.fill_between([2e-4, 1], p_val, 0.5, color="grey", alpha=0.025)
        i += 1

    # Title and annotations
    plt.title(r"$\bf{2016\,\, CMS\,\, Open\,\, Data\,\, DoubleMuon}$", loc="right", fontsize=10)

    ax.text(0.05, 0.4, r"\textbf{Opposite Sign} Muons", transform=ax.transAxes,
            fontsize=10, verticalalignment='top', alpha=0.75)
    ax.text(0.05, 0.35, r"Muon Iso\_04 $\geq$ 0.55", transform=ax.transAxes,
            fontsize=8, verticalalignment='top', alpha=0.75)
    ax.text(0.05, 0.30, r"8.7 fb$^{-1}$, $\sqrt{s} = 13$ TeV", transform=ax.transAxes,
            fontsize=8, verticalalignment='top', alpha=0.75)

    # plt.legend(loc="lower right", fontsize=7, ncol=1)

    # Legend: Fit type (color)
    legend_elements_color = [
        Line2D([0], [0], color=color, lw=6, label=label.capitalize())
        for label, color in fit_colors.items()
    ]

    # Legend: Bin width (linestyle)
    legend_elements_style = [
        Line2D([0], [0], color="black", linestyle=ls, lw=2.5, label=f"Binwidth: {bw:.1f}%")
        for bw, ls in bin_styles.items()
    ]

    legend1 = ax.legend(handles=legend_elements_color, loc="lower left", fontsize=9)
    legend2 = ax.legend(handles=legend_elements_style, loc="lower right", fontsize=9)
    ax.add_artist(legend1)

    if save_name is not None:
        for name in save_name:
            plt.savefig(name)

    return fig, ax


def bump_hunting(args, num_SR_bins=8, fit_degree=3):
    postfix = "" if not args.no_signal else "_no_signal"

    with open(args.config_workflow) as f:
        config = yaml.safe_load(f)
    step_dir = "step4_bump_hunting"
    inputdir = clean_and_append(config["output"]["storedir"], postfix)

    cwd = os.getcwd()
    base_dir = os.path.dirname(os.path.abspath(args.config_workflow))
    os.chdir(base_dir)
    global_config.load_yaml(config["train-cls"]["config"])
    with open(config["input"]["event_info"]) as f:
        event_info = yaml.safe_load(f)
    os.chdir(cwd)
    plotdir = os.path.join(clean_and_append(config["output"]["plotdir"], postfix), step_dir)
    os.makedirs(plotdir, exist_ok=True)


    # Get the data for the region
    # fpr_thresholds = [1.0, 0.1, 0.01, 0.001]
    fpr_thresholds = np.logspace(-3, 0, 31)[::-1]

    fit_result = None
    if args.plot_only:
        fit_result_json = args.summary_json
        with open(fit_result_json) as f:
            fit_result_ = json.load(f)
            fit_result = fit_result_.get(str(num_SR_bins), {}).get(str(fit_degree), {}).get("save_data", None)
            fit_result["bonus_significance"] = fit_result_.get(str(num_SR_bins), {}).get(str(fit_degree), {}).get("bonus_significance", None)


    if args.no_signal:
        channel = "OS" if args.test_no_signal else "SS"
    else:
        channel = "SS" if args.test_no_signal else "OS"


    if not args.test_no_signal:
        fname = os.path.join(clean_and_append(inputdir, "_score"), "final_results.csv")
        df_data = pd.read_csv(fname)

        print(f"✅ Loaded '{fname}' with {len(df_data)} rows.")
        save_data, bonus_sigificance = plot_histograms_with_fits(
            fpr_thresholds,
            data_dict=df_data,
            title="Toy Model Test",
            SR_left=config['mass-windows']['SR-left'],
            SR_right=config['mass-windows']['SR-right'],
            SB_left=config['mass-windows']['SB-left'],
            SB_right=config['mass-windows']['SB-right'],
            fit_degree= fit_degree,
            num_bins_SR=num_SR_bins,
            take_score_avg=True,
            ymin=1e-2,
            ymax=1e5,
            # legend_title = r"$\bf{Delphes: Z^\prime(300 GeV)}$",
            legend_title=r"$\bf{CMS Open Data}$",
            ncpu=len(fpr_thresholds),
            fit=True,
            fit_result = fit_result,
            channel = channel
        )
        plt.savefig(os.path.join(plotdir, f'bump_hunting_{num_SR_bins}bins_{fit_type[fit_degree]}_fit_xgb.png'))

        plot_significane_vs_pfr(
            fpr_thresholds = fpr_thresholds,
            save_data = save_data,
            num_bins_SR = num_SR_bins,
            fit_degree = fit_degree,
            bonus_significance= bonus_sigificance,
            ymin=1e-15,
            save_name = [os.path.join(plotdir, f"bump_hunting_{num_SR_bins}bins_{fit_type[fit_degree]}_xgb_pfr.pdf"), os.path.join(plotdir, f"bump_hunting_{num_SR_bins}bins_{fit_type[fit_degree]}_xgb_pfr.png")],
        )

    else:
        fname = os.path.join(clean_and_append(inputdir, "_score"), "final_results_no_signal.csv")
        df_data = pd.read_csv(fname)
        print(f"✅ Loaded '{fname}' with {len(df_data)} rows.")

        save_data, bonus_sigificance = plot_histograms_with_fits(
            fpr_thresholds,
            data_dict=df_data,
            title="Toy Model Test",
            SR_left=config['mass-windows']['SR-left'],
            SR_right=config['mass-windows']['SR-right'],
            SB_left=config['mass-windows']['SB-left'],
            SB_right=config['mass-windows']['SB-right'],
            fit_degree=fit_degree,
            num_bins_SR=num_SR_bins,
            take_score_avg=True,
            ymin=1e-2,
            ymax=1e5,
            # legend_title = r"$\bf{Delphes: Z^\prime(300 GeV)}$",
            legend_title=r"$\bf{CMS Open Data}$",
            ncpu=len(fpr_thresholds),
            fit=True,
            fit_result = fit_result,
            channel = channel
        )
        plt.savefig(os.path.join(plotdir, f'bump_hunting_{num_SR_bins}bins_{fit_type[fit_degree]}_fit_xgb_nosignal.png'))

        plot_significane_vs_pfr(
            fpr_thresholds=fpr_thresholds,
            save_data=save_data,
            num_bins_SR=num_SR_bins,
            fit_degree=fit_degree,
            bonus_significance=bonus_sigificance,
            ymin=1e-15,
            save_name=[os.path.join(plotdir, f"bump_hunting_{num_SR_bins}bins_{fit_type[fit_degree]}_xgb_pfr_nosignal.pdf"),
                       os.path.join(plotdir, f"bump_hunting_{num_SR_bins}bins_{fit_type[fit_degree]}_xgb_pfr_nosignal.png")],
        )
    return fpr_thresholds, save_data, bonus_sigificance

def run_bump_hunting(params):
    args, num_SR_bins, fit_degree = params
    fpr_thresholds, save_data, bonus_significance = bump_hunting(
        args,
        num_SR_bins=num_SR_bins,
        fit_degree=fit_degree
    )
    return num_SR_bins, fit_degree, fpr_thresholds, save_data, bonus_significance


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    parser.add_argument("--no_signal", action = "store_true", default = False)
    parser.add_argument("--test_no_signal", action = "store_true", default = False,)
    parser.add_argument("--plot_only", action = "store_true", default = False,)

    # Parse command-line arguments
    args = parser.parse_args()
    # Explore the provided HDF5 file

    postfix = "" if not args.no_signal else "_no_signal"

    with open(args.config_workflow) as f:
        config = yaml.safe_load(f)
    step_dir = "step4_bump_hunting"

    cwd = os.getcwd()
    base_dir = os.path.dirname(os.path.abspath(args.config_workflow))
    os.chdir(base_dir)
    global_config.load_yaml(config["train-cls"]["config"])
    with open(config["input"]["event_info"]) as f:
        event_info = yaml.safe_load(f)
    os.chdir(cwd)
    plotdir = os.path.join(clean_and_append(config["output"]["plotdir"], postfix), step_dir)
    if args.test_no_signal:
        postfix_for_plot = "_no_signal"
    else:
        postfix_for_plot = ""

    summary_json = os.path.join(plotdir, f"summary_data{postfix_for_plot}.json")
    args.summary_json = summary_json
    # bin_percent = {8: 2.3, 12: 1.5, 16: 1.1}
    # fit_type = {3: "cubic", 4: "quartic", 5: "quintic", 6: "sextic", 7: "septimic", 8: "octic"}

    summary_data = dict()

    # Create all combinations of parameters to evaluate
    param_grid = [(args, num_SR_bins, fit_degree) for num_SR_bins in bin_percent for fit_degree in fit_type]

    # Define a wrapper function to pass multiple arguments

    # Use ProcessPoolExecutor to parallelize
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(run_bump_hunting, param_grid)

    # Collect the results into summary_data
    for num_SR_bins, fit_degree, fpr_thresholds, save_data, bonus_significance in results:
        if num_SR_bins not in summary_data:
            summary_data[num_SR_bins] = dict()
        summary_data[num_SR_bins][fit_degree] = {
            "fpr_thresholds": fpr_thresholds,
            "save_data": save_data,
            "bonus_significance": bonus_significance
        }


    console = Console()
    channel = "OS" if not args.no_signal else "SS"
    if args.no_signal:
        test_channel = "OS" if args.test_no_signal else "SS"
    else:
        test_channel = "SS" if args.test_no_signal else "OS"

    table = Table(title=f"Bonus Significance Summary / train: {channel}, test: {test_channel} ", box=box.SIMPLE_HEAVY)
    table.add_column("Bin Width (%)", justify="right", style="cyan")
    table.add_column("Fit Type", justify="left", style="magenta")
    table.add_column("Bonus Significance", justify="left", style="green")

    for num_SR_bins, fit_dict in summary_data.items():
        for fit_degree, data in fit_dict.items():
            bonus = data.get("bonus_significance")
            if bonus is None:
                bonus_str = "[dim]None[/dim]"
            elif hasattr(bonus, "__len__") and not isinstance(bonus, str):
                bonus_str = ", ".join(f"{v:.4f}" for v in bonus)
            else:
                bonus_str = f"{bonus:.4f}"

            table.add_row(str(bin_percent[num_SR_bins]), str(fit_type[fit_degree]), bonus_str)
    console.print(table)
    def safe_converter(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    if args.test_no_signal:
        postfix_for_plot = "_no_signal"
    else:
        postfix_for_plot = ""

    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2, default=safe_converter)

    plot_significance_vs_fpr_all(
        summary_data,
        ymin=1e-15,
        save_name=[os.path.join(plotdir, f"bump_hunting_summary{postfix_for_plot}.pdf"),os.path.join(plotdir, f"bump_hunting_summary{postfix_for_plot}.png")],
        bin_percent=bin_percent,
        fit_type=fit_type,
    )


if __name__ == "__main__":
    main()

