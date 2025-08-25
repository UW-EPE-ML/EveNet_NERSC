import argparse
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
sns.set(style="whitegrid", context="paper", font_scale=1.2)
from matplotlib.gridspec import GridSpec

def plot_sig_eff_vs_bkg_eff(
    fit_dir,
    masses,
    sig_eff_list= [10, 25, 50, 75, 90],
    output_dir=None,
    filename_prefix=None,
    dataset_size_require = 1.0,
    show=False,
):
    """
    Plot signal efficiency vs background efficiency for each mass point.

    Args:
        fit_dir (str): Path to the directory containing subfolders with roc_results.npz.
        masses (list[str]): List of mass points as strings, e.g., ["15", "20", "25"].
        output_dir (str): Directory to store plots. Defaults to fit_dir/fit-summary.
        filename_prefix (str): Optional prefix for output plot filenames.
        show (bool): If True, display plots interactively.
    """
    def extract_metric(sr_data, key):
        parts = key.split(".")
        for part in parts:
            sr_data = sr_data[part]
        return sr_data

    training_dataset_size_absolute = 1000000
    roc_summaries = defaultdict(lambda: defaultdict(list))  # mass → base_legend → list of (bkg_eff, sig_eff)

    for legend in os.listdir(fit_dir):

        roc_results = os.path.join(fit_dir, legend, "summary", "roc_results.npz")
        if not os.path.isfile(roc_results):
            continue
        base_legend, dataset_size = parse_legend(legend)
        if not (str(dataset_size) == str(dataset_size_require)):
            continue
        roc_data = np.load(roc_results, allow_pickle=True)

        for m in masses:
            key = f"haa_ma{m}"
            if key not in roc_data:
                continue
            try:
                sr_data = roc_data[key].item()["SR"]
                for sig_eff in sig_eff_list:
                    metric_key = f"BackgroundRejection.bkg_rejection_at_{sig_eff}pct_signal"
                    unc_metric_key = f"BackgroundRejection-unc.bkg_rejection_at_{sig_eff}pct_signal"

                    value = extract_metric(sr_data, metric_key)
                    value_unc = extract_metric(sr_data, unc_metric_key)
                    if m not in roc_summaries:
                        roc_summaries[m] = {}
                    if base_legend not in roc_summaries[m]:
                        roc_summaries[m][base_legend] = []
                    roc_summaries[m][base_legend].append((sig_eff, value, value_unc))
            except Exception as e:
                print(f"Error processing {key} in {legend}: {e}")

        if output_dir is None:
            output_dir = os.path.join(fit_dir, "fit-summary")
        os.makedirs(output_dir, exist_ok=True)

        # Define colorblind-friendly palette
        custom_colors = sns.color_palette("colorblind", 6)
        palette_dict = {
            "spanet": custom_colors[0],
            "pretrain": custom_colors[1],
            "other": custom_colors[2],
        }

        for mass, legend_dict in sorted(roc_summaries.items(), key=lambda x: float(x[0])):
            fig = plt.figure(figsize=(8, 8))
            gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

            ax_main = fig.add_subplot(gs[0])
            ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

            if not legend_dict:
                continue

            ylabel = "Bkgd Rej. Rate"
            title = f"{ylabel} vs Sig Rate\n$m_a$ = {mass} GeV"

            # Collect data into dict for ease
            legend_data = {}
            ref_key = None

            for base_legend, sv_list in legend_dict.items():
                sv_list = sorted(sv_list, key=lambda x: x[0])
                dataset_sizes = [x[0] for x in sv_list]
                values = [x[1] for x in sv_list]
                unc_vals = [x[2] if x[2] is not None else 0 for x in sv_list]
                legend_data[base_legend] = (dataset_sizes, values, unc_vals)

                if "spanet" in base_legend and "assignment-on" not in base_legend:
                    ref_key = base_legend

            if ref_key is None:
                print(
                    f"Warning: No reference legend (spanet + assignment-on) found for mass {mass}. Skipping ratio plot.")
                continue

            # Reference values
            ref_xs, ref_vals, ref_uncs = legend_data[ref_key]

            for base_legend, (xs, vals, uncs) in sorted(legend_data.items()):
                # Determine color by model type
                if 'spanet' in base_legend:
                    model_type = "spanet"
                elif 'pretrain' in base_legend:
                    model_type = "pretrain"
                else:
                    model_type = "other"
                color = palette_dict.get(model_type, palette_dict["other"])

                # Linestyle by assignment flag
                is_solid = 'assignment-on' in base_legend
                alpha_fill = 0.1 if is_solid else 0.05
                hatch = None if is_solid else '///'

                model_name = base_legend.split('-')[0]
                pretrain_on = 'pretrain' in base_legend
                assignment_on = 'assignment-on' in base_legend
                segmentation_on = 'segmentation-on' in base_legend
                legend_name = "SPANet" if model_name == "spanet" else "EveNet"
                if not (model_name == "spanet"):
                    if pretrain_on:
                        legend_name += "-f.t."
                    else:
                        legend_name += "-Scratch"

                if assignment_on and not segmentation_on:
                    legend_name += "(w Assignment)"
                    linestyle = "-"
                    markerstyle = "o"

                elif segmentation_on and not assignment_on:
                    legend_name += "(w Segmentation)"
                    linestyle = ":"
                    markerstyle = "o"

                elif assignment_on and segmentation_on:
                    legend_name += "(w Assignment + Segmentation)"
                    linestyle = "-."
                    markerstyle = "x"
                else:
                    linestyle = '--'
                    markerstyle = "o"

                # --- Main plot ---
                ax_main.plot(xs, vals, label=legend_name, color=color,
                             linestyle=linestyle, marker=markerstyle)
                if any(uncs):
                    lower = [v - u for v, u in zip(vals, uncs)]
                    upper = [v + u for v, u in zip(vals, uncs)]
                    if hatch:
                        ax_main.fill_between(xs, lower, upper, facecolor='none',
                                             edgecolor=color, hatch=hatch, linewidth=0.0, alpha=0.5)
                    else:
                        ax_main.fill_between(xs, lower, upper, color=color, alpha=alpha_fill)

                # --- Ratio plot ---
                ref_dict = dict(zip(ref_xs, zip(ref_vals, ref_uncs)))
                ratios = []
                ratio_uncs = []
                for x, v, u in zip(xs, vals, uncs):
                    if x not in ref_dict:
                        ratios.append(np.nan)
                        ratio_uncs.append(0)
                        continue
                    v_ref, u_ref = ref_dict[x]
                    if v_ref == 0:
                        ratios.append(np.nan)
                        ratio_uncs.append(0)
                        continue
                    ratio = v / v_ref
                    rel_unc = np.sqrt((u / v) ** 2 + (u_ref / v_ref) ** 2) if v > 0 else 0
                    ratios.append(ratio)
                    ratio_uncs.append(ratio * rel_unc)

                ax_ratio.plot(xs, ratios, color=color, linestyle=linestyle, marker=markerstyle)
                if any(ratio_uncs):
                    lower = [r - u for r, u in zip(ratios, ratio_uncs)]
                    upper = [r + u for r, u in zip(ratios, ratio_uncs)]
                    if hatch:
                        ax_ratio.fill_between(xs, lower, upper, facecolor='none',
                                              edgecolor=color, hatch=hatch, linewidth=0.0, alpha=0.5)
                    else:
                        ax_ratio.fill_between(xs, lower, upper, color=color, alpha=alpha_fill)

            # --- Format axes ---
            ax_main.set_ylabel(ylabel)
            # ax_main.set_title(title)
            # ax_main.legend(loc="best", title=None)
            ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            plt.setp(ax_main.get_xticklabels(), visible=False)

            ax_ratio.axhline(1.0, color='gray', linestyle='--')
            ax_ratio.set_ylabel("Ratio vs SPANet")
            ax_ratio.set_xlabel("Signal Efficiency (%)")
            ax_ratio.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

            # ax_main.set_xscale("log")
            ax_main.set_ylim(5e-1, None)
            ax_main.set_yscale("log")
            # ax_ratio.set_xscale("log")
            # --- Save plot ---
            plt.tight_layout()
            filename = f"{filename_prefix + '_' if filename_prefix else ''}{ylabel.replace(' ', '_')}_vs_SigEff_mass{mass}_withRatio_dataset_size{dataset_size_require}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            filename = f"{filename_prefix + '_' if filename_prefix else ''}{ylabel.replace(' ', '_')}_vs_SigEff_mass{mass}_withRatio_dataset_size{dataset_size_require}.pdf"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            if show:
                plt.show()
            else:
                plt.close()

def plot_metric_vs_dataset_size(
    fit_dir,
    masses,
    metric_key="AUC",
    output_dir=None,
    filename_prefix=None,
    show=False,
    unc_metrics = None
):
    """
    Plot a given metric vs dataset size for each mass point.

    Args:
        fit_dir (str): Path to the directory containing subfolders with roc_results.npz.
        masses (list[str]): List of mass points as strings, e.g., ["15", "20", "25"].
        metric_key (str): Metric to plot. Examples:
                          - "AUC"
                          - "BackgroundRejection.bkg_rejection_at_70pct_signal"
        output_dir (str): Directory to store plots. Defaults to fit_dir/fit-summary.
        filename_prefix (str): Optional prefix for output plot filenames.
        show (bool): If True, display plots interactively.
    """
    def extract_metric(sr_data, key):
        parts = key.split(".")
        for part in parts:
            sr_data = sr_data[part]
        return sr_data

    training_dataset_size_absolute = 1000000
    roc_summaries = defaultdict(lambda: defaultdict(list))  # mass → base_legend → list of (dataset_size, metric)

    for legend in os.listdir(fit_dir):
        roc_results = os.path.join(fit_dir, legend, "summary", "roc_results.npz")
        if not os.path.isfile(roc_results):
            continue
        base_legend, dataset_size = parse_legend(legend)
        roc_data = np.load(roc_results, allow_pickle=True)

        for m in masses:
            key = f"haa_ma{m}"
            if key not in roc_data:
                continue
            try:
                sr_data = roc_data[key].item()["SR"]
                value = extract_metric(sr_data, metric_key)
                if unc_metrics is not None:
                    value_unc = extract_metric(sr_data, unc_metrics)
                if m not in roc_summaries:
                    roc_summaries[m] = {}
                if base_legend not in roc_summaries[m]:
                    roc_summaries[m][base_legend] = []
                roc_summaries[m][base_legend].append((float(dataset_size)*training_dataset_size_absolute, value, value_unc if unc_metrics else None))
            except Exception as e:
                print(f"Error processing {key} in {legend}: {e}")

    if output_dir is None:
        output_dir = os.path.join(fit_dir, "fit-summary")
    os.makedirs(output_dir, exist_ok=True)

    # Define colorblind-friendly palette
    custom_colors = sns.color_palette("colorblind", 6)
    palette_dict = {
        "spanet": custom_colors[0],
        "pretrain": custom_colors[1],
        "other": custom_colors[2],
    }

    for mass, legend_dict in sorted(roc_summaries.items(), key=lambda x: float(x[0])):
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        if not legend_dict:
            continue

        ylabel = metric_key.replace("BackgroundRejection.", "Bkgd Rej. Rate @ ") \
            .replace("bkg_rejection_at_", r"$\epsilon_{sig}$ = ") \
            .replace("pct_signal", "%")
        title = f"{ylabel} vs Dataset Size\n$m_a$ = {mass} GeV"

        # Collect data into dict for ease
        legend_data = {}
        ref_key = None

        for base_legend, sv_list in legend_dict.items():
            sv_list = sorted(sv_list, key=lambda x: x[0])
            dataset_sizes = [x[0] for x in sv_list]
            values = [x[1] for x in sv_list]
            unc_vals = [x[2] if x[2] is not None else 0 for x in sv_list]
            legend_data[base_legend] = (dataset_sizes, values, unc_vals)

            if "spanet" in base_legend and "assignment-on" not in base_legend:
                ref_key = base_legend

        if ref_key is None:
            print(f"Warning: No reference legend (spanet + assignment-on) found for mass {mass}. Skipping ratio plot.")
            continue

        # Reference values
        ref_xs, ref_vals, ref_uncs = legend_data[ref_key]

        for base_legend, (xs, vals, uncs) in sorted(legend_data.items()):
            # Determine color by model type
            if 'spanet' in base_legend:
                model_type = "spanet"
            elif 'pretrain' in base_legend:
                model_type = "pretrain"
            else:
                model_type = "other"
            color = palette_dict.get(model_type, palette_dict["other"])

            # Linestyle by assignment flag
            is_solid = 'assignment-on' in base_legend
            alpha_fill = 0.1 if is_solid else 0.05
            hatch = None if is_solid else '///'

            model_name = base_legend.split('-')[0]
            pretrain_on = 'pretrain' in base_legend
            assignment_on = 'assignment-on' in base_legend
            segmentation_on = 'segmentation-on' in base_legend
            legend_name = "SPANet" if model_name == "spanet" else "EveNet"
            if not (model_name == "spanet"):
                if pretrain_on:
                    legend_name += "-f.t."
                else:
                    legend_name += "-Scratch"

            if assignment_on and not segmentation_on:
                legend_name += "(w Assignment)"
                linestyle = "-"
                markerstyle = "o"

            elif segmentation_on and not assignment_on:
                legend_name += "(w Segmentation)"
                linestyle = ":"
                markerstyle = "o"

            elif assignment_on and segmentation_on:
                legend_name += "(w Assignment + Segmentation)"
                linestyle = "-."
                markerstyle = "x"
            else:
                linestyle = '--'
                markerstyle = "o"

            # --- Main plot ---
            ax_main.plot(xs, vals, label=legend_name, color=color,
                         linestyle=linestyle, marker=markerstyle)
            if any(uncs):
                lower = [v - u for v, u in zip(vals, uncs)]
                upper = [v + u for v, u in zip(vals, uncs)]
                if hatch:
                    ax_main.fill_between(xs, lower, upper, facecolor='none',
                                         edgecolor=color, hatch=hatch, linewidth=0.0, alpha=0.5)
                else:
                    ax_main.fill_between(xs, lower, upper, color=color, alpha=alpha_fill)

            # --- Ratio plot ---
            ref_dict = dict(zip(ref_xs, zip(ref_vals, ref_uncs)))
            ratios = []
            ratio_uncs = []
            for x, v, u in zip(xs, vals, uncs):
                if x not in ref_dict:
                    ratios.append(np.nan)
                    ratio_uncs.append(0)
                    continue
                v_ref, u_ref = ref_dict[x]
                if v_ref == 0:
                    ratios.append(np.nan)
                    ratio_uncs.append(0)
                    continue
                ratio = v / v_ref
                rel_unc = np.sqrt((u / v) ** 2 + (u_ref / v_ref) ** 2) if v > 0 else 0
                ratios.append(ratio)
                ratio_uncs.append(ratio * rel_unc)

            ax_ratio.plot(xs, ratios, color=color, linestyle=linestyle, marker=markerstyle)
            if any(ratio_uncs):
                lower = [r - u for r, u in zip(ratios, ratio_uncs)]
                upper = [r + u for r, u in zip(ratios, ratio_uncs)]
                if hatch:
                    ax_ratio.fill_between(xs, lower, upper, facecolor='none',
                                          edgecolor=color, hatch=hatch, linewidth=0.0, alpha=0.5)
                else:
                    ax_ratio.fill_between(xs, lower, upper, color=color, alpha=alpha_fill)

        # --- Format axes ---
        ax_main.set_ylabel(ylabel)
        # ax_main.set_title(title)
        # ax_main.legend(loc="best", title=None)
        ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ax_ratio.axhline(1.0, color='gray', linestyle='--')
        ax_ratio.set_ylabel("Ratio vs SPANet")
        ax_ratio.set_xlabel("Dataset Size")
        ax_ratio.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        ax_main.set_xscale("log")
        ax_ratio.set_xscale("log")
        # --- Save plot ---
        plt.tight_layout()
        filename = f"{filename_prefix + '_' if filename_prefix else ''}{ylabel.replace(' ', '_')}_vs_DatasetSize_mass{mass}_withRatio.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        filename = f"{filename_prefix + '_' if filename_prefix else ''}{ylabel.replace(' ', '_')}_vs_DatasetSize_mass{mass}_withRatio.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

def extract_mass_key_values(json_data):
    mass_values = []
    y_values = []

    for key, values in json_data.items():
        match = re.match(r"haa_ma(\d+)", key)
        if match:
            mass = int(match.group(1))
            y = values[2]  # 3th value
            mass_values.append(mass)
            y_values.append(y)

    # Sort by mass
    sorted_pairs = sorted(zip(mass_values, y_values))
    return zip(*sorted_pairs)  # returns (masses, y_values)

def parse_legend(legend):
    """
    Extract base legend and dataset size from legend string.

    Example:
    legend = "myLegend_dataset_size100"
    returns ("myLegend", 100)
    """
    match = re.match(r"(.+)-dataset_size([0-9]*\.?[0-9]+)", legend)
    if match:
        base_legend = match.group(1)
        dataset_size = str(match.group(2))
        return base_legend, dataset_size
    else:
        return legend, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_dir", required=True, help="Base path to store_dir")
    args = parser.parse_args()

    fit_dir = os.path.join(args.store_dir, "fit")

    plt.figure(figsize=(20, 10))

    for legend in os.listdir(fit_dir):
        summary_path = os.path.join(fit_dir, legend, "summary", "ExpectedLimit.json")
        if not os.path.isfile(summary_path):
            continue  # skip         # roc_results = os.path.join(fit_dir, legend, "summary", "roc_results.npz")
        # if os.path.isfile(roc_results):
        #     roc_data = np.load(roc_results, allow_pickle=True)
        #     print(roc_data)
        #     for m, v in zip(masses, values):
        #         if m not in roc_summaries:
        #             roc_summaries[m] = []
        #         roc_summaries[m][base_legend].append(roc_data[f"haa_ma{m}"]["SR"])if ExpectedLimit.json is missing

        with open(summary_path) as f:
            data = json.load(f)

        masses, values = extract_mass_key_values(data)
        if masses:
            plt.plot(masses, values, marker="o", label=legend)

    plt.xlabel("m(a)[GeV]")
    plt.ylabel("Expected Limit")
    plt.title("Expected Limits vs MASS")
    plt.grid(True)
    plt.legend(title=None)
    plt.tight_layout()
    plt.show()

    # Save the plot
    os.makedirs(os.path.join(args.store_dir, "fit-summary"), exist_ok=True)
    output_path = os.path.join(args.store_dir, "fit-summary", "ExpectedLimits_vs_MASS.png")
    plt.savefig(output_path)

    # --- New: Plot ExpectedLimit vs dataset_size, grouped by mass point ---
    # Structure: {mass: {base_legend: list of (dataset_size, value)}}


    mass_plots = {}
    roc_summaries = {}
    all_masses = []

    for legend in os.listdir(fit_dir):
        summary_path = os.path.join(fit_dir, legend, "summary", "ExpectedLimit.json")
        if not os.path.isfile(summary_path):
            continue

        base_legend, dataset_size = parse_legend(legend)
        print(f"Processing legend: {base_legend}, dataset_size: {dataset_size}")
        if dataset_size is None:
            continue  # skip legends without dataset size info

        with open(summary_path) as f:
            data = json.load(f)

        masses, values = extract_mass_key_values(data)
        all_masses = masses
        for m, v in zip(masses, values):
            if m not in mass_plots:
                mass_plots[m] = {}
            if base_legend not in mass_plots[m]:
                mass_plots[m][base_legend] = []
            mass_plots[m][base_legend].append((dataset_size, v))




    # Now create one plot per mass point
    for mass, legend_dict in sorted(mass_plots.items()):
        plt.figure()

        # Find the base_legend with the longest sv_list first
        longest_legend = max(legend_dict.items(), key=lambda item: len(item[1]))[0]

        # Plot the longest one first
        sv_list = sorted(legend_dict[longest_legend], key=lambda x: x[0])
        dataset_sizes, values = zip(*sv_list)
        color = 'blue' if 'pretrain' in longest_legend else 'orange'
        if 'spanet' in longest_legend:
            color = 'red'
        linestyle = '-' if 'assignment-on' in longest_legend else '--'
        plt.plot(dataset_sizes, values, marker='o', label=longest_legend,
                 color=color, linestyle=linestyle)

        # Plot the rest (excluding the longest one)
        for base_legend, sv_list in legend_dict.items():
            if base_legend == longest_legend:
                continue
            sv_list = sorted(sv_list, key=lambda x: x[0])
            dataset_sizes, values = zip(*sv_list)
            color = 'blue' if 'pretrain' in base_legend else 'orange'
            if 'spanet' in base_legend:
                color = 'red'
            linestyle = '-' if 'assignment-on' in base_legend else '--'
            plt.plot(dataset_sizes, values, marker='o', label=base_legend,
                     color=color, linestyle=linestyle)
  
        plt.xlabel("Dataset Size")
        plt.ylabel("Expected Limit")
        plt.title(f"Expected Limit vs Dataset Size at m(a) = {mass} GeV")
        plt.grid(True)
        plt.legend(title = None)
        plt.tight_layout()
        plot_path = os.path.join(args.store_dir, "fit-summary", f"ExpectedLimits_vs_DatasetSize_mass{mass}.png")
        plt.savefig(plot_path)
        plt.show()

    plot_metric_vs_dataset_size(
        fit_dir,
        all_masses,
        metric_key="AUC",
        output_dir=os.path.join(args.store_dir, "fit-summary"),
        filename_prefix=None,
        show=False
    )

    for pct in [10, 25, 30, 50, 70, 75, 90]:
        metrics = "BackgroundRejection.bkg_rejection_at_{}pct_signal".format(pct)
        plot_metric_vs_dataset_size(
            fit_dir,
            all_masses,
            metric_key=metrics,
            output_dir=os.path.join(args.store_dir, "fit-summary"),
            filename_prefix=None,
            show=False,
            unc_metrics="BackgroundRejection-unc.bkg_rejection_at_{}pct_signal".format(pct)
        )

    for dataset_size in [0.01, 0.03, 0.1, 0.3, 1.0]:
        plot_sig_eff_vs_bkg_eff(
            fit_dir,
            all_masses,
            sig_eff_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            output_dir=os.path.join(args.store_dir, "fit-summary"),
            filename_prefix=None,
            show=False,
            dataset_size_require = dataset_size
        )



if __name__ == "__main__":
    main()
