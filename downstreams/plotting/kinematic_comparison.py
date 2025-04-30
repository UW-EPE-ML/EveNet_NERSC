import numpy as np
import colorsys

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects


def compute_kde(values, positions, grid_size):
    """ Computes KDE on a given grid for parallel processing """
    kde = gaussian_kde(values, bw_method=0.1)
    return kde(positions).reshape(grid_size, grid_size)  # Correctly reshaped


def parallel_contour_plot(ax, kin, truth_kin, kin_range, grid_size, contour_colors, c_percent):
    """ Optimized parallel contour density plot using Grid-Based KDE """
    X, Y = np.meshgrid(
        np.linspace(kin_range[0], kin_range[1], grid_size),
        np.linspace(kin_range[0], kin_range[1], grid_size)
    )

    positions = np.vstack([X.ravel(), Y.ravel()])

    # Parallel KDE computation
    kde_results = Parallel(n_jobs=-1)(
        delayed(compute_kde)(np.vstack([k, tk]), positions, grid_size)
        for k, tk in zip(kin, truth_kin)
    )

    # Plot contours

    # c_percent = np.array([10, 25, 75, 100])  # Define % levels for enclosed data
    # Define % levels for enclosed data
    for i, Z in enumerate(kde_results):
        Z_flat = Z.ravel()  # Flatten KDE density values
        sorted_Z = np.sort(Z_flat)[::-1]  # Sort densities in descending order

        # **Compute cumulative probability mass (CDF of KDE)**
        cumsum_Z = np.cumsum(sorted_Z)
        cumsum_Z /= cumsum_Z[-1]  # Normalize to [0, 1]

        # **Find contour levels that enclose `c_percent`% of the total data**
        levels = np.interp(1 - (c_percent / 100.0), cumsum_Z, sorted_Z)
        # **Plot contours using the correct levels**
        ax.contourf(X, Y, Z, levels=levels, cmap=contour_colors[i], alpha=0.5, norm=mcolors.LogNorm())
        contour_lines = ax.contour(X, Y, Z, levels=levels, colors=[contour_colors[i](1.0)], linewidths=0.5, alpha=0.5)
        enclosed_labels = {lvl: f"{100 - p}%" for lvl, p in zip(levels, c_percent)}

        # Get RGB from colormap
        base_color = contour_colors[i](1.0)
        rgb = base_color[:3]  # remove alpha if present

        # Convert to HLS using colorsys
        h, l, s = colorsys.rgb_to_hls(*rgb)

        # Darken by reducing lightness
        darker_rgb = colorsys.hls_to_rgb(h, l * 0.5, s)
        darker_rgb_with_alpha = (*darker_rgb, 1.0)

        dr = kin_range[1] - kin_range[0]
        low = kin_range[0] + 0.25 * (i + 1) * dr
        high = kin_range[0] + (0.2 * (i + 1) + 0.1) * dr
        random_range = np.random.uniform(low, high)

        positions = [(random_range, random_range)]
        texts = ax.clabel(
            contour_lines, inline=False, fontsize=22, fmt=lambda x: enclosed_labels.get(x, ""),
            colors=[darker_rgb],
            manual=positions,
        )

        for text in texts:
            # text.set_fontweight('bold')
            text.set_path_effects([
                path_effects.withStroke(linewidth=0.5, foreground=darker_rgb)
            ])


def plot_kinematics_comparison(
        axs, kin, truth_kin, bins=50, kin_range=None,
        xlabel="Kinematics", ylabel="Counts",
        ratio_label=None,
        labels=None, colors=None, contour_colors=None,
        normalize_col=False, log_z=False,log_y=False,
        c_percent = np.array([10, 25, 75, 100]),
):
    if ratio_label is None:
        ratio_label = ["Reco", "Truth"]
    if kin_range is None:
        all_kin = np.concatenate(kin + truth_kin)
        kin_range = (np.min(all_kin), np.max(all_kin))

    bin_edges = np.linspace(kin_range[0], kin_range[1], bins + 1)

    if colors is None:
        colors = ['blue', 'red']  # Colors for histograms and ratio plots
    if contour_colors is None:
        contour_colors = [
            LinearSegmentedColormap.from_list(f"CustColor{i}", ["white", colors[i]])
            for i in range(len(colors))
        ]  # Contour colormaps

    # Upper panel: Overlaid histograms
    ax1 = axs[0]
    y_max = 0
    for i, (k, tk) in enumerate(zip(kin, truth_kin)):
        kin_hist, bin_edges = np.histogram(k, bins=bin_edges)
        truth_hist, _ = np.histogram(tk, bins=bin_edges)

        label = labels[i] if labels else f"Dataset {i + 1}"

        # Reco Histogram (Solid Line, Transparent)
        ax1.plot(
            bin_edges[:-1], kin_hist, drawstyle="steps-mid", color=colors[i], linewidth=2, alpha=1.0,
            label=f"{label} (Reco)"
        )

        # Truth Histogram (Dashed Line, No Transparency)
        ax1.plot(
            bin_edges[:-1], truth_hist, drawstyle="steps-mid", linestyle="--", color=colors[i], linewidth=2, alpha=0.75,
            label=f"{label} (Truth)"
        )

        y_max = max(y_max, kin_hist.max(), truth_hist.max())

    ax1.set_ylabel(ylabel)
    if log_y:
        ax1.set_yscale('log')
    else:
        ax1.set_ylim(0, 2.0 * y_max)

    # ** Create Custom Legend Handles **
    # 1️⃣ Global Reco/Truth Styles
    line_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label=ratio_label[0]),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label=ratio_label[1]),
    ]

    # 2️⃣ Particle Colors
    color_handles = [Patch(facecolor=colors[i], edgecolor=colors[i], label=labels[i]) for i in range(len(kin))]
    color_handles += line_handles

    # ** Add Legends **
    ax1.legend(handles=color_handles, loc="upper right", frameon=False)

    # Middle panel: Overlaid ratio plot
    ax2 = axs[1]
    for i, (k, tk) in enumerate(zip(kin, truth_kin)):
        kin_hist, _ = np.histogram(k, bins=bin_edges)
        truth_hist, _ = np.histogram(tk, bins=bin_edges)

        # Compute ratio and error safely
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(kin_hist, truth_hist, out=np.zeros_like(kin_hist, dtype=float), where=truth_hist != 0)
            ratio_err = np.zeros_like(ratio)
            nonzero_mask = (kin_hist > 0) & (truth_hist > 0)
            ratio_err[nonzero_mask] = ratio[nonzero_mask] * np.sqrt(
                1 / kin_hist[nonzero_mask] + 1 / truth_hist[nonzero_mask])

        # Define upper/lower bounds for error band
        ratio_upper = ratio + ratio_err
        ratio_lower = ratio - ratio_err

        # Plot central ratio line
        ax2.plot(
            bin_edges[:-1], ratio, marker='.', color=colors[i], markersize=8, alpha=0.7,
            label=labels[i] if labels else f"Dataset {i + 1}"
        )

        # Fill between for error band
        ax2.fill_between(bin_edges[:-1], ratio_lower, ratio_upper, color=colors[i], alpha=0.3)

    ax2.set_ylabel("/".join(ratio_label))
    ax2.axhline(y=1, color='gray', linestyle='--')  # Reference line
    # ax2.set_ylim(0.5, 1.5)
    ax2.set_ylim(0.0, 2.0)

    # Lower panel: Fast Contour Density Plot using Grid-Based KDE
    ax3 = axs[2]

    if len(kin) > 1 and len(truth_kin) > 1:
        grid_size = 50  # Define grid size for KDE evaluation
        parallel_contour_plot(ax3, kin, truth_kin, kin_range, grid_size, contour_colors, c_percent)
        # ax3.grid(True)

    else:
        # **Alternative: Direct 2D Histogram for Reference**
        k, tk = kin[0], truth_kin[0]
        # **Find a common range for square bins**
        k_min, k_max = np.min(k), np.max(k)
        tk_min, tk_max = np.min(tk), np.max(tk)
        common_min = max(k_min, tk_min, kin_range[0])  # Ensure it includes kin_range
        common_max = min(k_max, tk_max, kin_range[1])

        # **Define square bins using the common range**
        bins = np.linspace(common_min, common_max, num=bins)  # Ensure square binning

        # **Compute 2D histogram**
        H, xedges, yedges = np.histogram2d(k, tk, bins=[bins, bins])  # Square bins

        if normalize_col:
            H_sum = H.sum(axis=1, keepdims=True)  # Sum along rows for each column
            H = np.divide(H, H_sum, where=H_sum != 0)  # Avoid division by zero

        # **Plot heatmap with log scaling**
        ax3.pcolormesh(
            xedges, yedges, H.T, cmap=contour_colors[0], alpha=0.95,
            norm=mcolors.LogNorm() if log_z else None,
        )

    ax3.plot([kin_range[0], kin_range[1]], [kin_range[0], kin_range[1]], 'r--', label='y=x')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(f'{ratio_label[1]} {xlabel}')
    ax3.set_xlim(kin_range)
    ax3.set_ylim(kin_range)
