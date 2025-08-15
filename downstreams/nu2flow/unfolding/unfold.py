#!/usr/bin/env python
import numpy as np
import pandas as pd
from pathlib import Path
import uproot

import ROOT

ROOT.gSystem.Load(
    "/Users/avencastmini/PycharmProjects/EveNet/downstreams/nu2flow/unfolding/RooUnfold/build/libRooUnfold.dylib")


def read_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def hist_setup():
    bin_edges = {
        "m_tt": [0, 400, 500, 800, np.inf],
        "dphi_ll": [0.0, 0.25, 0.5, 0.75, 1.0],
        "pt_t1": [0, 75, 125, 175, np.inf],
        "pt_t2": [0, 75, 125, 175, np.inf],
        "pt_tt": [0, 70, 140, 200, np.inf],
        "y_tt": [-np.inf, -1.0, 0.0, 1.0, np.inf],
    }

    return bin_edges


def pre_bin(df_all, bin_edges, recon_types=None):
    if recon_types is None:
        recon_types = ["truthnu", "prednu"]

    df_new = pd.DataFrame()

    for var_short, edges in bin_edges.items():
        truth_col = f"{var_short}_truth"
        for reco_type in recon_types:
            reco_col = f"{var_short}_reco_{reco_type}"

            # Bin both columns directly into a new column
            df_new[f"{var_short}_truth"] = pd.cut(df_all[truth_col], bins=edges, labels=False)
            df_new[f"{var_short}_{reco_type}"] = pd.cut(df_all[reco_col], bins=edges, labels=False)

    mtt_bin_numbers = len(bin_edges["m_tt"]) - 1
    for recon_type in recon_types:
        for var, edges in bin_edges.items():
            if var == "m_tt": continue

            truth_mtt_col = "m_tt_truth"
            reco_mtt_col = f"m_tt_{recon_type}"

            truth_col = f"{var}_truth"
            reco_col = f"{var}_{recon_type}"

            # merged bins is:
            # b = truth_col + truth_mtt_col * ( len(edges) - 1 )
            df_new[f"{var}_truth_final"] = df_new[truth_col] + df_new[truth_mtt_col] * mtt_bin_numbers
            df_new[f"{var}_{recon_type}_final"] = df_new[reco_col] + df_new[reco_mtt_col] * mtt_bin_numbers

    return df_new


def build_response(df_in: pd.DataFrame, truth_col: str, recon_col: str, bin_nums: int = 16):
    response = ROOT.RooUnfoldResponse(bin_nums, -0.5, bin_nums - 0.5)

    for row in df_in.itertuples():
        truth_val = getattr(row, truth_col)
        reco_val = getattr(row, recon_col)

        if truth_val >= 0 and reco_val >= 0:
            response.Fill(reco_val, truth_val)
        elif truth_val >= 0:
            response.Miss(truth_val)

    return response


def build_histograms(df_in: pd.DataFrame, truth_col: str, recon_col: str, bin_nums: int = 16):
    h_truth = ROOT.TH1D(f"h_{truth_col}", "Truth", bin_nums, -0.5, bin_nums - 0.5)
    h_reco = ROOT.TH1D(f"h_{recon_col}", "Reco", bin_nums, -0.5, bin_nums - 0.5)

    for row in df_in.itertuples():
        truth_val = getattr(row, truth_col)
        reco_val = getattr(row, recon_col)

        if truth_val >= 0:
            h_truth.Fill(truth_val)
        if reco_val >= 0:
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


def main(df: pd.DataFrame, plot_path: Path = None):
    # plot_path.mkdir(parents=True, exist_ok=True)

    bin_edges = hist_setup()

    df_new = pre_bin(df, bin_edges)

    df_all = []
    for category in ["truthnu", "prednu"]:
        for var in bin_edges.keys():
            if var == "m_tt": continue

            df_in = df_new

            response = build_response(
                df_in=df_in,
                truth_col=f"{var}_truth_final", recon_col=f"{var}_{category}_final"
            )

            h_truth, h_measure = build_histograms(
                df_in=df_in,
                truth_col=f"{var}_truth_final", recon_col=f"{var}_{category}_final"
            )

            unfold = ROOT.RooUnfoldSvd(response, h_measure, 7)
            hUnfold = unfold.Hunfold(2)

            # unfold.PrintTable(ROOT.cout, h_truth)
            # plot_histograms(h_truth, h_measure, hUnfold, save_path=plot_path / f"{var}_{category}.pdf")

            df_truth = hist_to_df(h_truth, name=f"{var}_{category}_truth", nbins=16)
            df_reco = hist_to_df(h_measure, name=f"{var}_{category}_reco", nbins=16)
            df_unfold = hist_to_df(hUnfold, name=f"{var}_{category}_unfold", nbins=16)

            df_combined = pd.concat([df_truth, df_reco, df_unfold], axis=1)
            df_all.append(df_combined)

            print(f"Category: {category} - Variable: {var} - done")

    df_all = pd.concat(df_all, axis=1)

    return df_all


if __name__ == '__main__':
    df = read_csv(Path("/Users/avencastmini/PycharmProjects/EveNet/downstreams/nu2flow/aux/df_all.csv"))
    plot_path = Path("/Users/avencastmini/PycharmProjects/EveNet/downstreams/nu2flow/aux/unfolding")

    main(df=df, plot_path=plot_path)