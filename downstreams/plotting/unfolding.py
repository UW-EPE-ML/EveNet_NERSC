from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def plot_uncertainty_with_ratio(
        mtt_labels,
        pt_labels,
        xlabel,
        methods,
        ratio_baseline_name,
        p_dir: Path,
        save_name=None,
        ratio_baseline_min: float = -1.0,
        ratio_baseline_max: float = 1.0,
        ratio_y_label: str = "Ratio to baseline",
):
    def pad_step_data(data):
        data = np.array(data)
        padded_data = np.concatenate([[data[0]], data, [data[-1]]])
        padded_x = np.arange(-1, len(data) + 1)
        return padded_x, padded_data

    num_blocks = len(mtt_labels)
    bins_per_block = len(pt_labels)
    total_bins = num_blocks * bins_per_block
    x = np.arange(total_bins)
    x_labels = pt_labels * num_blocks

    # Find baseline method
    baseline_method = next((m for m in methods if m["name"] == ratio_baseline_name), None)
    if baseline_method is None:
        raise ValueError(f"Baseline method '{ratio_baseline_name}' not found in methods.")

    # Create figure
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.0}
    )

    # --- Top panel: Uncertainty ---
    for method in methods:
        x_pad, y_pad = pad_step_data(method["data"])
        ax_top.step(x_pad, y_pad, where='mid', label=method["name"], color=method["color"], linewidth=2)

    for i in range(bins_per_block, total_bins, bins_per_block):
        ax_top.axvline(i - 0.5, color='black', linestyle='--', lw=1)

    max_val = max([max(m["data"]) for m in methods])
    for i, label in enumerate(mtt_labels):
        center = i * bins_per_block + bins_per_block / 2 - 0.5
        ax_top.text(center, max_val * 1.05, label, ha='center', fontsize=12)

    ax_top.set_ylabel("Relative uncertainty")
    ax_top.set_ylim(1.0, max_val * 1.15)
    ax_top.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 1.12),
        frameon=False,
        ncol=len(methods),
        fontsize=12,
        handlelength=1.5,
        columnspacing=1.0
    )

    # --- Bottom panel: Ratio ---
    max_ratio = -np.inf
    min_ratio = np.inf
    for method in methods:
        if method["name"] == ratio_baseline_name:
            continue
        ratio = - (method["data"] - baseline_method["data"]) / (baseline_method["data"] + 1e-6)

        r_x_pad, r_y_pad = pad_step_data(ratio)
        ax_bot.step(r_x_pad, r_y_pad, where='mid', color=method["color"], linewidth=2)

        for j, val in enumerate(ratio):
            if val < ratio_baseline_min:
                ax_bot.plot(x[j], ratio_baseline_min, marker='v', color=method["color"], markersize=8)
            elif val > ratio_baseline_max:
                ax_bot.plot(x[j], ratio_baseline_max, marker='^', color=method["color"], markersize=8)

        max_ratio = max(max_ratio, ratio.max())
        min_ratio = min(min_ratio, ratio.min())


    ax_bot.axhline(0.0, color='black', lw=1)
    ax_bot.set_ylabel(ratio_y_label)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels, rotation=45, ha='right')
    ax_bot.set_ylim(max(ratio_baseline_min, min_ratio * 1.05), min(ratio_baseline_max, max_ratio * 1.15))
    print("min_ratio, max_ratio", min_ratio, max_ratio)

    for i in range(bins_per_block, total_bins, bins_per_block):
        ax_bot.axvline(i - 0.5, color='black', linestyle='--', lw=1)

    ax_bot.set_xlabel(xlabel)
    ax_bot.set_xlim(-0.5, total_bins - 0.5)
    # set log scale for y-axis
    # ax_bot.set_yscale('log')

    plt.tight_layout()
    if save_name:
        if not os.path.exists(p_dir / "uncertainty"):
            os.makedirs(p_dir / "uncertainty")
        plt.savefig(p_dir / "uncertainty" / save_name)
        plt.close()
    else:
        # Show the plot
        plt.show()


# Generalized plotting function
def plot_block_response(
        response,
        var_labels,
        mtt_labels,
        title=None,
        xlabel="Truth $m_{tt}$ bin",
        ylabel="Reco variable bin",
        p_dir: Path = None,
        save_name=None,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    truth_sums = response.sum(axis=0, keepdims=True)
    normed = 100 * np.divide(response, truth_sums, where=truth_sums != 0)

    im = ax.imshow(normed, origin='lower', cmap='Blues', vmin=0, vmax=100)

    # Annotate matrix
    for i in range(normed.shape[0]):
        for j in range(normed.shape[1]):
            val = normed[i, j]
            if val > 1:
                ax.text(j, i, f"{val:.0f}", ha='center', va='center', fontsize=7)

    # Grid lines
    block_size = len(var_labels)
    for i in range(0, response.shape[0], block_size):
        ax.axhline(i - 0.5, color='k', linestyle='--', lw=1)
        ax.axvline(i - 0.5, color='k', linestyle='--', lw=1)
    ax.axhline(response.shape[0] - 0.5, color='k', linestyle='--', lw=1)
    ax.axvline(response.shape[1] - 0.5, color='k', linestyle='--', lw=1)

    # X ticks: 1 per mtt bin (block center)
    xticks = [i * block_size + block_size / 2 - 0.5 for i in range(len(mtt_labels))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(mtt_labels, fontsize=10)

    # Y ticks: 1 per var bin
    yticks = list(range(block_size * len(mtt_labels)))
    ytick_labels = var_labels * len(mtt_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)  # shrink width to 3%
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Migration [%]", fontsize=9)

    # Trace fraction
    trace = np.trace(response)
    total = response.sum()
    trace_frac = trace / total if total > 0 else 0
    ax.text(1.0, 1.02, f"trace fraction = {trace_frac:.2f}", transform=ax.transAxes,
            ha='right', fontsize=10)

    plt.tight_layout()
    # plt.show()
    if save_name:
        if not os.path.exists(p_dir / "response"):
            os.makedirs(p_dir / "response")
        plt.savefig(p_dir / "response" / save_name)
    plt.close()
