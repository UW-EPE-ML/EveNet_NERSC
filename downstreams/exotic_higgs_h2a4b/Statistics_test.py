import optparse, argparse
import os, sys

sys.path.insert(0, 'util')
from common import CheckDir, read_json, fig_save_and_close, read_yaml, store_json, weighted_roc_curve
from plot_tool import plot_data_mc
import cabinetry
import json
import pyhf
import ROOT
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
import iminuit
from termcolor import cprint
from sklearn.metrics import roc_curve, auc
import numpy as np

cprint("iminuit version: {}".format(iminuit.__version__), "green")


def extract_hist_data(hist):
    bin_contents = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)])
    bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, hist.GetNbinsX() + 2)])
    return bin_contents, bin_edges


def Generate_Test_Data(signal, config):
    workspace = os.path.join(config.outdir, signal)
    CheckDir(workspace)

    file_reference = ROOT.TFile.Open(config.sourceFile, "READ")

    h_pseudodata = None

    Histograms = file_reference.GetListOfKeys()
    Histograms_ToWrite = dict()
    for Hist in Histograms:
        cross_section = None
        for bkg_ in config.Process["Background"]:
            if Hist.GetName().startswith(bkg_ + "_"): cross_section = config.Process["Background"][bkg_]["xsec"]
        if Hist.GetName().startswith(sig_ + "_"): cross_section = config.Process["Signal"][signal]["xsec"]

        if cross_section is None: continue

        # scale = 0.1 if Hist.GetName().startswith(sig_ + "_") else 1
        scale = 1

        print(Hist.GetName(), config.Lumi * 1000)
        Histograms_ToWrite[Hist.GetName()] = file_reference.Get(Hist.GetName()).Clone()
        Histograms_ToWrite[Hist.GetName()].SetDirectory(0)
        Histograms_ToWrite[Hist.GetName()].Scale(config.Lumi * scale)
        Histograms_ToWrite[Hist.GetName()].SetName(Hist.GetName())
        print(Histograms_ToWrite[Hist.GetName()].Integral())
    file_reference.Close()

    ###########################
    ##  Generate Pseudodata  ##
    ###########################

    config_yaml = read_yaml(config.config_yml)

    for region in config_yaml['Regions']:
        region_ = region['RegionPath']
        for bkg_idx, bkg_ in enumerate(config.Process["Background"]):
            if (bkg_idx == 0):
                Histograms_ToWrite["Pseudodata_{}_{}".format(config.observable, region_)] = Histograms_ToWrite[
                    "{}_{}_{}".format(bkg_, config.observable, region_)].Clone()
                Histograms_ToWrite["Pseudodata_{}_{}".format(config.observable, region_)].SetDirectory(0)
                Histograms_ToWrite["Pseudodata_{}_{}".format(config.observable, region_)].SetName(
                    "Pseudodata_{}_{}".format(config.observable, region_))

            else:
                Histograms_ToWrite["Pseudodata_{}_{}".format(config.observable, region_)].Add(
                    Histograms_ToWrite["{}_{}_{}".format(bkg_, config.observable, region_)].Clone())

    file_output = ROOT.TFile.Open(os.path.join(workspace, "analysis_ntuple_original.root"), "RECREATE")
    file_output.cd()
    for Histogram_Name in Histograms_ToWrite:
        Histograms_ToWrite[Histogram_Name].Write()
    file_output.Close()


    Histograms_ToAnalyze = dict()
    file_output = ROOT.TFile.Open(os.path.join(workspace, "analysis_ntuple.root"), "RECREATE")
    file_output.cd()
    for Histogram_Name in Histograms_ToWrite:
        Histograms_ToAnalyze[Histogram_Name] = Histograms_ToWrite[Histogram_Name].Clone()
        Histograms_ToAnalyze[Histogram_Name].Rebin(50) # 1000 bins to 20 bins
        Histograms_ToAnalyze[Histogram_Name].Write()
        Histograms_ToAnalyze[Histogram_Name].SetDirectory(0)
    file_output.Close()


    #########################
    ##  Plot Distribution  ##
    #########################

    Edges = dict()
    Contents = dict()
    for region in config_yaml['Regions']:
        region_ = region['RegionPath']
        Contents[region_] = dict()
        for bkg_ in config.Process["Background"]:
            contents, edge = extract_hist_data(Histograms_ToAnalyze["{}_{}_{}".format(bkg_, config.observable, region_)])
            Edges[region_] = edge
            Contents[region_][bkg_] = {"Type": "Background", "Content": contents, "Yield": np.sum(contents)}

        contents, edge = extract_hist_data(Histograms_ToAnalyze["{}_{}_{}".format(signal, config.observable, region_)])
        Contents[region_][signal] = {"Type": "Signal", "Content": contents, "Yield": np.sum(contents)}

        contents, edge = extract_hist_data(
            Histograms_ToAnalyze["{}_{}_{}".format("Pseudodata", config.observable, region_)])
        Contents[region_]["Pseudodata"] = {"Type": "Data", "Content": contents, "Yield": np.sum(contents)}

    plot_data_mc(Edges, Contents, config, figure_path=os.path.join(workspace, "Distribution_log.png"),
                 close_figure=True, log_scale=True)
    plot_data_mc(Edges, Contents, config, figure_path=os.path.join(workspace, "Distribution.png"), close_figure=True)
    print("Distribution plots saved to: {}".format(workspace))



    Edges = dict()
    Contents = dict()

    for region in config_yaml['Regions']:
        region_ = region['RegionPath']
        Contents[region_] = dict()
        for bkg_ in config.Process["Background"]:
            contents, edge = extract_hist_data(Histograms_ToWrite["{}_{}_{}".format(bkg_, config.observable, region_)])
            Edges[region_] = edge
            Contents[region_][bkg_] = {"Type": "Background", "Content": contents, "Yield": np.sum(contents)}

        contents, edge = extract_hist_data(Histograms_ToWrite["{}_{}_{}".format(signal, config.observable, region_)])
        Contents[region_][signal] = {"Type": "Signal", "Content": contents, "Yield": np.sum(contents)}

        contents, edge = extract_hist_data(Histograms_ToWrite["{}_{}_{}".format("Pseudodata", config.observable, region_)])
        Contents[region_]["Pseudodata"] = {"Type": "Data", "Content": contents, "Yield": np.sum(contents)}

    # Now compute ROC and AUC for each region:
    ROC_results = dict()

    for region in Contents:
        signal_hist = Contents[region][signal]["Content"]
        # Sum all backgrounds into one
        bkg_hist = Contents[region]["Pseudodata"]["Content"]

        edges = Edges[region]
        bin_centers = (edges[:-1] + edges[1:]) / 2

        # Concatenate signal and background bin centers as scores
        y_scores = np.concatenate([bin_centers, bin_centers])
        # Labels: 1 for signal bins, 0 for background bins
        y_true = np.concatenate([np.ones_like(bin_centers), np.zeros_like(bin_centers)])
        # Weights are the bin contents (can be float)
        sample_weights = np.concatenate([signal_hist, bkg_hist])

        fpr, tpr, fpr_unc = weighted_roc_curve(y_true, y_scores, sample_weights, edges)

        # Compute ROC with weights
        # fpr, tpr, thresholds = roc_curve(y_true, y_scores, sample_weight=sample_weights)
        roc_auc = auc(fpr, tpr)

        # Background rejection at given signal efficiency WPs
        wps = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
        bkg_rejections = {}
        bkg_rejections_unc = {}
        for wp in wps:
            idx = np.argmin(np.abs(tpr - wp))
            bkg_rej = 1.0 / fpr[idx] if fpr[idx] > 0 else np.inf
            bkg_rejections[f"bkg_rejection_at_{int(wp * 100)}pct_signal"] = bkg_rej
            bkg_rejections_unc[f"bkg_rejection_at_{int(wp * 100)}pct_signal"] =  fpr_unc[idx] * (bkg_rej * bkg_rej)


        ROC_results[region] = {
            "AUC": roc_auc,
            "FPR-unc": fpr_unc,
            "FPR": fpr,
            "TPR": tpr,
            "BackgroundRejection": bkg_rejections,
            "BackgroundRejection-unc": bkg_rejections_unc
        }

    return ROC_results

# ROC_results now contains ROC info, AUC, and background rejection for each region



def plot_summary(Limit, config, cls_target=0.3):
    plotdir = os.path.join(config.outdir, 'summary')
    CheckDir(plotdir)

    fig, ax = plt.subplots(layout="constrained")

    mass_bin = []
    Limit_Result = dict()

    for signal_ in Limit:
        Mass = config.Process["Signal"][signal_]["Mass"]
        mass_bin.append(Mass)
        Limit_Result[Mass] = Limit[signal_]

    mass_bin = np.array(mass_bin)
    mass_bin = np.sort(mass_bin)

    xmin = min(mass_bin)
    xmax = max(mass_bin)

    observed_limit = np.array([Limit_Result[mass_].observed_limit for mass_ in mass_bin])
    expected_limit = np.array([Limit_Result[mass_].expected_limit for mass_ in mass_bin])

    ymin = min(min(expected_limit[:, 0]) * 0.9, 0.01)
    ymax = max(max(expected_limit[:, 4]) * 1.1, 0.5)

    # Observed limit
    #  ax.plot(mass_bin, observed_limit, "o-", color="black", label=r"observed 95% CL")

    # Expected limit
    ax.plot(mass_bin, expected_limit[:, 2], "--", color="black", label=r"expcted 95% CL")

    # 1 and 2 sigma bands
    ax.fill_between(
        mass_bin,
        expected_limit[:, 1],
        expected_limit[:, 3],
        color="limegreen",
        label=r"expected 95% CL $\pm 1\sigma$"
    )

    ax.fill_between(
        mass_bin,
        expected_limit[:, 0],
        expected_limit[:, 4],
        color="yellow",
        label=r"expected 95% CL $\pm 2\sigma$",
        zorder=0
    )

    cls_pct_is_integer = math.isclose(cls_target * 100, round(cls_target * 100))
    cls_label = f"{cls_target:.{0 if cls_pct_is_integer else 2}%}"
    # line through CLs = cls_target
    ax.hlines(
        cls_target,
        xmin=xmin,
        xmax=xmax,
        linestyle="dashdot",
        color="red",
        label="Br = " + cls_label,  # 2 decimals unless they are both 0, then 0
        zorder=1,  # draw beneath observed / expected
    )

    # increase font sizes
    for item in (
            [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize("large")

    # minor ticks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.legend(frameon=False, fontsize='large')
    ax.set_xlabel(r"$m_a$[GeV]")
    ax.set_ylabel(r"95%$CL_s$ on $Br(h\rightarrow aa \rightarrow 4b)$")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(axis="both", which="major", pad=8)
    ax.tick_params(direction='in', top=True, right=True, which="both")
    plt.text(0.7, 0.65, r"Lumi = {} / fb".format(int(config.Lumi / 1000)), transform=ax.transAxes, fontsize='large')
    fig_save_and_close(fig, os.path.join(plotdir, 'summary.pdf'), True)
    return fig


def Statistics_test(signal, config):
    #####################
    ##  Basic setting  ##
    #####################
    workspace = os.path.join(config.outdir, signal)
    CheckDir(workspace)
    config_yml = os.path.join(workspace, 'statistics_config.yml')
    os.system('cp {template} {file}'.format(template=config.config_yml, file=config_yml))

    #############################
    ##  Configuration setting  ##
    #############################

    os.system('sed -i -e "s|OUTDIR|{outdir}|g" {file}'.format(file=config_yml, outdir=workspace))
    os.system('sed -i -e "s|HISTOGRAMROOT|{hist_root}|g" {file}'.format(file=config_yml,
                                                                        hist_root=os.path.join(workspace,
                                                                                               "analysis_ntuple.root")))
    os.system('sed -i -e "s|VARIABLE|{observable}|g" {file}'.format(file=config_yml, observable=config.observable))
    os.system('sed -i -e "s|SIGNALNAME|{signal}|g" {file}'.format(file=config_yml, signal=signal))

    ##############################
    ##  Create model workspace  ##
    ##############################

    config_ = cabinetry.configuration.load(config_yml)
    cabinetry.templates.collect(config_)
    ws = cabinetry.workspace.build(config_)
    cabinetry.workspace.save(ws, os.path.join(workspace, 'workspace.json'))

    ############
    ##  Pull  ##
    ############
    model, data = cabinetry.model_utils.model_and_data(ws)
    fit_results = cabinetry.fit.fit(model, data)
    pull_fig = cabinetry.visualize.pulls(
        fit_results, exclude="mu", close_figure=True, save_figure=True, figure_folder=workspace
    )

    ranking_results = cabinetry.fit.ranking(model, data)
    pull_impact = cabinetry.visualize.ranking(
        ranking_results, close_figure=True, save_figure=True, figure_folder=workspace
    )

    ##############
    ##  Prefit  ##
    ##############

    model_prediction = cabinetry.model_utils.prediction(model)
    figs = cabinetry.visualize.data_mc(model_prediction, data, close_figure=True, figure_folder=workspace,
                                       save_figure=True)

    ###############
    ##  Postfit  ##
    ###############
    model_prediction_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
    figs = cabinetry.visualize.data_mc(model_prediction_postfit, data, close_figure=True, figure_folder=workspace,
                                       save_figure=True)
    cabinetry.tabulate.yields(model_prediction_postfit, data, save_tables=True, table_folder=workspace)

    ####################
    ##  Significance  ##
    ####################

    significance_result = cabinetry.fit.significance(model, data, poi_name="mu")
    print("Significance: ", significance_result)

    #############
    ##  Limit  ##
    #############

    limit_result = cabinetry.fit.limit(model, data, confidence_level=0.95, poi_name="mu", bracket=(0.001, 1.0))
    limit_fig = cabinetry.visualize.limit(limit_result, close_figure=True, save_figure=True, figure_folder=workspace)

    return limit_result, significance_result


if __name__ == '__main__':

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--Lumi', default=300, type=float, help='In fb^{-1} unit')
    parser.add_argument('--signal', default=['all'], type=str, nargs='+',
                        help='List of signals, by default use all from event_config/process.json')
    parser.add_argument('--process_json', type=str, default='event_config/process.json')
    parser.add_argument('--sourceFile', type=str)
    parser.add_argument('--observable', type=str, default='Count')
    parser.add_argument('--config_yml', type=str, default='event_config/Statistics_Test.yml')
    parser.add_argument('--outdir', type=str, default='Limit_study')
    parser.add_argument('--log_scale_x', action='store_true')
    parser.add_argument('--log_scale', action='store_true')
    config = parser.parse_args()

    config.Lumi = config.Lumi * 1000  # From fb^-1 to pb^-1
    config.Process = read_json(config.process_json)
    if 'all' in config.signal:
        sig_list = []
        for sig_ in config.Process["Signal"]:
            sig_list.append(sig_)
        config.signal = sig_list

    CheckDir(config.outdir)

    Limit = dict()
    Significance = dict()
    ExpectedLimit = dict()
    ROCCurve = dict()
    observable_template = config.observable

    for sig_ in config.signal:
        try:
            print("running {sig}...".format(sig=sig_))
            config.observable = observable_template.replace('SIGNAL', sig_).replace("MASS", str(
                config.Process["Signal"][sig_]["Mass"]))
            ROCCurve[sig_] = Generate_Test_Data(signal=sig_, config=config)
            Limit[sig_], Significance[sig_] = Statistics_test(signal=sig_, config=config)
            print(Limit[sig_].expected_limit)
            ExpectedLimit[sig_] = Limit[sig_].expected_limit.tolist()
        except Exception as e:
            print("Error processing signal {}: {}".format(sig_, e))
            continue

    # print(Significance)
    plot_summary(Limit, config)
    store_json(ExpectedLimit, os.path.join(config.outdir, 'summary', 'ExpectedLimit.json'))
    print(ROCCurve)
    np.savez(os.path.join(config.outdir, 'summary', "roc_results.npz"),
             **ROCCurve)
