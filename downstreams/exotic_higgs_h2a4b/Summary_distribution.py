import argparse
import pathlib
import uproot
import matplotlib.pyplot as plt
import re
import numpy as np


def parse_conditions(path):
    """
    Extract conditions from the directory name.
    Example:
    spanet-pretrain-assignment-on-segmentation-dataset_size1.0/ntuple.root
    """
    name = path.parent.name
    parts = name.split('-')

    conditions = {
        "network": parts[0],
        "train": parts[1],
        "assignment": parts[3],
        "segmentation": "segmentation" in name,
        "dataset_size": re.search(r"dataset_size([\d\.]+)", name).group(1)
    }
    return conditions


def load_hist(file_path, mass=125, nbins=None, normalize=True):
    """Load and optionally rebin histogram for a given mass from ROOT file."""
    with uproot.open(file_path) as f:
        hist_bkg = f[f"QCD_MVAscore{mass}_SR"]
        hist_sig = f[f"haa_ma{mass}_MVAscore{mass}_SR"]
        values_bkg = hist_bkg.values()
        values_sig = hist_sig.values()
        edges = hist_bkg.axis().edges()

    if nbins is not None:
        # Define new edges
        new_edges = np.linspace(edges[0], edges[-1], nbins + 1)

        # Digitize original bin centers
        centers = (edges[:-1] + edges[1:]) / 2
        rebinned_bkg = np.zeros(nbins)
        rebinned_sig = np.zeros(nbins)
        indices = np.digitize(centers, new_edges) - 1

        # Sum content into new bins
        for i, idx in enumerate(indices):
            if 0 <= idx < nbins:
                rebinned_bkg[idx] += values_bkg[i]
                rebinned_sig[idx] += values_sig[i]

        values_bkg, values_sig, edges = rebinned_bkg, rebinned_sig, new_edges

    if normalize:
        total_bkg = values_bkg.sum()
        total_sig = values_sig.sum()
        if total_bkg > 0:
            values_bkg = values_bkg / total_bkg
        if total_sig > 0:
            values_sig = values_sig / total_sig
    return values_bkg, values_sig, edges

def main():
    parser = argparse.ArgumentParser(description="Plot signal/background distributions from ROOT files")
    parser.add_argument("directory", type=str, help="Base directory containing ntuple.root files")
    parser.add_argument("--mass", type=int, default=125, help="Mass hypothesis (default=125)")
    parser.add_argument("--output", type=str, default="dist_comparison.png", help="Output plot filename")
    args = parser.parse_args()

    base_dir = pathlib.Path(args.directory)
    files = list(base_dir.glob("**/ntuple.root"))

    if not files:
        print("No ROOT files found!")
        return

    plt.figure(figsize=(8, 6))

    # Some distinguishable styles
    # Background (cold)
    bkg_colors = ['#1f77b4',  # blue
                  '#17becf',  # cyan
                  '#2ca02c']  # green

    # Signal (warm)
    sig_colors = ['#d62728',  # red
                  '#ff7f0e',  # orange
                  '#bcbd22']  # yellow/golden

    linestyles = ["-", "--", "-.", ":"]

    for i, file_path in enumerate(sorted(files)):
        try:
            vals_bkg, vals_sig, edges = load_hist(file_path, args.mass, nbins=20)
            conds = parse_conditions(file_path)
            if not (conds['dataset_size'] == "1.0"):
                continue

            if conds['network'] == "spanet":
                bkg_color = bkg_colors[0]
                sig_color = sig_colors[0]
                label = "SPANet"
            elif f"{conds['network']}-{conds['train']}" == "evenet-pretrain":
                bkg_color = bkg_colors[1]
                sig_color = sig_colors[1]
                label = "EveNet-f.t."
            else:
                bkg_color = bkg_colors[2]
                sig_color = sig_colors[2]
                label = "EveNet-scratch"

            if conds['assignment'] == "on" and conds['segmentation']:
                label += " (w assignment + segmentation)"
                linestyle = "-."
                marker_style = "x"
            elif conds['assignment'] == "off" and conds['segmentation']:
                label += " (w segmentation)"
                linestyle = ":"
                marker_style = "o"
            elif conds['assignment'] == "on" and not conds['segmentation']:
                label += " (w assignment)"
                linestyle = "-"
                marker_style = "o"
            else:
                linestyle = "--"
                marker_style = "o"

            centers = 0.5 * (edges[1:] + edges[:-1])
            plt.plot(
                centers, vals_bkg,
                label=f"{label}-sig",
                color=bkg_color,
                linestyle=linestyle,
                linewidth=1.8
            )
            plt.plot(
                centers, vals_sig,
                label=f"{label}-sig",
                color=sig_color,
                linestyle=linestyle,
                linewidth=1.8
            )
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    plt.xlabel(r"$m_{aa}$ [GeV]")
    plt.ylabel("Events / bin")
    plt.title(f"Signal/Background distributions (m={args.mass} GeV)")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot as {args.output}")


if __name__ == "__main__":
    main()
