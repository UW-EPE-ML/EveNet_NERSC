#!/usr/bin/env python
import numpy as np
import pandas as pd
from pathlib import Path
import uproot

import ROOT

ROOT.gSystem.Load(
    "/Users/avencastmini/PycharmProjects/EveNet/downstreams/nu2flow/unfolding/RooUnfold/build/libRooUnfold.dylib")


def pre_bin(df_all, bin_edges, recon_types=None):
    if recon_types is None:
        recon_types = ["recon"]

    df_new = pd.DataFrame()

    for var_short, edges in bin_edges.items():
        truth_col = f"{var_short}_truth"
        for reco_type in recon_types:
            reco_col = f"{var_short}_{reco_type}"

            # Bin both columns directly into a new column
            df_new[f"{var_short}_truth"] = pd.cut(df_all[truth_col], bins=edges, labels=False)
            df_new[f"{var_short}_{reco_type}"] = pd.cut(df_all[reco_col], bins=edges, labels=False)

    for recon_type in recon_types:
        for var, edges in bin_edges.items():
            if var == "m_tt": continue

            var_bin_numbers = len(edges) - 1

            truth_mtt_col = "m_tt_truth"
            reco_mtt_col = f"m_tt_{recon_type}"

            truth_col = f"{var}_truth"
            reco_col = f"{var}_{recon_type}"

            # merged bins is:
            # b = truth_col + truth_mtt_col * ( len(edges) - 1 )
            df_new[f"{var}_truth_final"] = df_new[truth_col] + df_new[truth_mtt_col] * var_bin_numbers
            df_new[f"{var}_{recon_type}_final"] = df_new[reco_col] + df_new[reco_mtt_col] * var_bin_numbers

    return df_new


def build_response(df_in: pd.DataFrame, truth_col: str, recon_col: str, bin_nums: int = 16):
    response = ROOT.RooUnfoldResponse(bin_nums, -0.5, bin_nums - 0.5)

    for row in df_in.itertuples():
        truth_val = getattr(row, truth_col)
        reco_val = getattr(row, recon_col)

        if truth_val != np.nan and reco_val != np.nan:
            response.Fill(reco_val, truth_val)
        elif truth_val != np.nan:
            response.Miss(truth_val)

    return response


def build_histograms(df_in: pd.DataFrame, truth_col: str, recon_col: str, bin_nums: int = 16):
    h_truth = ROOT.TH1D(f"h_{truth_col}", "Truth", bin_nums, -0.5, bin_nums - 0.5)
    h_reco = ROOT.TH1D(f"h_{recon_col}", "Reco", bin_nums, -0.5, bin_nums - 0.5)

    for row in df_in.itertuples():
        truth_val = getattr(row, truth_col)
        reco_val = getattr(row, recon_col)

        if truth_val != np.nan:
            h_truth.Fill(truth_val)
        if reco_val != np.nan:
            h_reco.Fill(reco_val)

    return h_truth, h_reco


def plot_histograms(h_truth, h_reco, hUnfold, save_path: Path):
    canvas = ROOT.TCanvas("RooUnfold", "SVD")
    ROOT.gStyle.SetOptStat(0)

    hUnfold.Draw()
    h_reco.Draw("SAME")
    h_reco.SetLineColor(2)
    h_truth.SetLineColor(8)
    h_truth.Draw("SAME")

    legend = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)  # (x1, y1, x2, y2) in NDC
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)  # transparent

    legend.AddEntry(hUnfold, "Unfolded", "p")
    legend.AddEntry(h_reco, "Reco (before unfolding)", "l")
    legend.AddEntry(h_truth, "Truth", "l")

    legend.Draw()

    if save_path:
        canvas.SaveAs(str(save_path))


def hist_to_df(hist, name="hist", nbins=16):
    data = {
        f"{name}_content": [],
        f"{name}_error": [],
    }

    for i in range(1, nbins + 1):  # ROOT bins are 1-indexed
        content = hist.GetBinContent(i)
        error = hist.GetBinError(i)

        data[f"{name}_content"].append(content)
        data[f"{name}_error"].append(error)

    return pd.DataFrame(data)


def main(full_df: pd.DataFrame, bin_edges: dict[str, np.ndarray]):
    plot_path = Path("plots/unfolding")
    plot_path.mkdir(parents=True, exist_ok=True)

    df_new = pre_bin(full_df, bin_edges)

    df_all = []
    for category in ["recon"]:
        for var in bin_edges.keys():
            if var == "m_tt": continue

            bin_nums = (len(bin_edges[var]) - 1) * (len(bin_edges["m_tt"]) - 1)  # total number of bins

            df_in = df_new

            response = build_response(
                df_in=df_in,
                truth_col=f"{var}_truth_final", recon_col=f"{var}_{category}_final",
                bin_nums=bin_nums
            )

            h_truth, h_measure = build_histograms(
                df_in=df_in,
                truth_col=f"{var}_truth_final", recon_col=f"{var}_{category}_final",
                bin_nums=bin_nums
            )

            unfold = ROOT.RooUnfoldSvd(response, h_measure, 5)
            hUnfold = unfold.Hunfold()

            # unfold.PrintTable(ROOT.cout, h_truth)
            plot_histograms(h_truth, h_measure, hUnfold, save_path=plot_path / f"{var}_{category}.pdf")

            df_truth = hist_to_df(h_truth, name=f"{var}_{category}_truth", nbins=bin_nums)
            df_reco = hist_to_df(h_measure, name=f"{var}_{category}_reco", nbins=bin_nums)
            df_unfold = hist_to_df(hUnfold, name=f"{var}_{category}_unfold", nbins=bin_nums)

            df_combined = pd.concat([df_truth, df_reco, df_unfold], axis=1)
            df_all.append(df_combined)

            print(f"Category: {category} - Variable: {var} - done")

    df_all = pd.concat(df_all, axis=1)

    return df_all
