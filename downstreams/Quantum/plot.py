import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def plot_poi(
        df: pd.DataFrame,
        model_cols: list[str],  # e.g., ['Concurrence', 'Bell']
        channel_col: str,
        model_configs: dict[str, dict[str, str]],
        channels_configs: dict[str, dict[str, str]],
        figsize=(15, 8),
        save_path: str = None,

        tick_fontsize=15,
        text_fontsize=16,
        text_precision=2,

        method_configs=None
):
    channels = [ch for ch in channels_configs]
    one = len(channels) == 1

    # Generate shifts for multiple significance columns
    def generate_shifts(n: int):
        return np.linspace(-(n - 1) / (2 * n), (n - 1) / (2 * n), n) / (1.05 if one else 1.25)

    shifts = generate_shifts(len(model_cols))
    y_values = np.arange(len(channels) + 0.5)

    # Create figure and broken axis (left: near-zero, right: main range)
    fig, ax_bottom_right = plt.subplots(
        1, 1, figsize=figsize
    )

    # Color configuration
    # Calculate dynamic axis ranges for the right panels
    x_max_text = 0.045
    x_text_axis = ax_bottom_right
    axis_range = {key: [-0.02, x_max_text] for key in ['bottom', 'top']}

    ax_bottom_right.set_xlim(*axis_range['bottom'])
    ax_bottom_right.set_xlabel('Uncertainty', fontsize=text_fontsize)

    x_text_shift = abs(axis_range['bottom'][0] - axis_range['bottom'][1]) * 0.025

    # Plotting loop
    for i, ch in enumerate(channels):
        if ch not in channels_configs:
            print(f"Warning: Channel '{ch}' not found in channels_configs. Skipping.")
            continue


        ch_data = df[df[channel_col] == ch]

        # draw a dashed line
        if i < len(channels) - 1:
            ax_bottom_right.axhline(y=i + 0.5, color='grey', linestyle='--', linewidth=0.5)

        for j, col in enumerate(model_cols):
            x_value = 0.0

            # Determine axis based on configuration
            axis_main = ax_bottom_right

            # Collect values for annotation
            up_full = float(ch_data[f'{col}_up'].iloc[0])
            down_full = float(ch_data[f'{col}_down'].iloc[0])

            marker_kwargs = dict(
                fmt='o', markersize=3, capsize=3.5, capthick=1.0, elinewidth=1.5, zorder=10
                # fmt = 'o', markersize = 5, capsize = 3, capthick = 1.75, elinewidth = 1.0, zorder = 10
            )

            # Plot statistical and systematic error bars
            axis_main.errorbar(
                x_value, y_values[i] + shifts[j],
                xerr=np.array([[down_full], [up_full]]),
                # color=color_axis_config[significance_configs[col]['axis']],
                color=method_configs[col.split('_')[-1]]['color'],
                label=f'{col} (Full)',
                **marker_kwargs,
            )

            var_text = rf"${model_configs[col]['label']}_{{{channels_configs[ch]['label'].replace('$', '')}}}"
            if model_configs[col].get('special', None):
                var_text = rf"${model_configs[col]['label']}"

            x_value = ch_data[f'{col}_value'].iloc[0]
            if channels_configs[ch].get('special', False):
                x_value = x_value - 1

            if np.isnan(x_value):
                # label = var_text + r"~~=~~-$"
                label = ""
            else:
                percentage_value =  down_full / abs(x_value)
                if percentage_value < 0.1:
                    percentage_str = f"{percentage_value * 100:.{text_precision}f}"  # Always two decimal places for single-digit numbers
                else:
                    percentage_str = f"-"

                label = (
                        # var_text +
                        rf"$~~\mathbf{{ {x_value:<8.{text_precision}f} }} "
                        rf"\mathbf{{ ~~~^{{+{up_full * 100:<10.{text_precision}f}}}_{{-{down_full * 100:<10.{text_precision}f}}} }} "
                        # rf"~~^{{+{up_stat:<7.{text_precision}f}}}_{{-{down_stat:<7.{text_precision}f}}} " +
                        # rf"~~^{{+{up_syst:<7.{text_precision}f}}}_{{-{down_syst:<7.{text_precision}f}}}" +
                        # "$"
                        rf"~~~~\mathbf{{{percentage_str}}}"
                    # rf"~~~\mathbf{{> 5 \sigma}}" +
                    # "$"
                )
                # label += f"~~~\mathbf{{-}}" if not channels_configs[ch].get('special', False) else f"~~~~{1 / percentage_value:.1f}\sigma"
                label += "$"

            # Annotate text to the right of the points (e.g., starting at x=1.2)
            x_text_axis.text(
                x_max_text - x_text_shift,  # Fixed x-position for annotations
                y_values[i] + shifts[j],
                label,
                fontsize=text_fontsize + 1, va='center', ha='right',
                # color=color_axis_config[significance_configs[col]['axis']],
                color=method_configs[col.split('_')[-1]]['color'],
            )

    # Construct text label
    shift = text_precision - 1
    sigma_str = r"{\sigma~[\%]}"
    label = (
            rf"${' ':^8s} " + f"{''.join(['~'] * shift)}" +
            f"{''.join(['~'] * (shift + 3))}" + r"Unc. [\%] " + f"{''.join(['~'] * (shift + 1 ))}" +
            f"{''.join(['~'] * (shift - 1))} {sigma_str:~^8s} {''.join(['~'] * (0))}" +
            "$"
    )

    # Annotate text to the right of the points (e.g., starting at x=1.2)
    x_text_axis.text(
        x_max_text - x_text_shift,  # Fixed x-position for annotations
        (y_values[-2] + 1.2 * shifts[-1] if one else y_values[-2] + 2.5 * shifts[-1]),
        label,
        fontsize=text_fontsize, color='black', va='center', ha='right'
    )
    # Change Y-axis ticks to channel names
    ax_bottom_right.set_yticks(y_values)
    if one:
        ax_bottom_right.set_yticklabels([])
        ax_bottom_right.set_ylim(y_values[0] - 0.5, y_values[-2] + 1.0)
    else:
        ax_bottom_right.set_yticklabels([channels_configs[ch]['label'] for ch in channels_configs] + [' '],
                                        fontsize=tick_fontsize)
        ax_bottom_right.set_ylim(y_values[0] - 0.5, y_values[-1] + 0.5)

    ax_bottom_right.tick_params(axis='both', labelsize=12)

    # Legends
    legend_patches = [
        mpatches.Patch(color=cfg['color'], label=cfg['label']) for key, cfg in method_configs.items()
        if any(key in s for s in model_cols)
    ]
    ax_bottom_right.legend(
        handles=legend_patches, loc='upper right', fontsize=text_fontsize,
        frameon=False, bbox_to_anchor=(1.0, 1.0)
    )

    # Extra Text
    lines = [
        r"$pp \to t\bar{t} \to b\bar{b}\ell\nu\ell\nu$",
        r"$13~\mathrm{TeV},\ 139~\mathrm{fb}^{-1}$",
    ]

    if one:
        lines += [r"Combined: $\pi\pi, \pi\rho, \rho\rho, \ell\pi, \ell\rho$"]
        y_start = 0.96
        y_step = 0.075
    else:
        y_start = 0.97
        y_step = 0.05
    for i, line in enumerate(lines):
        ax_bottom_right.text(
            0.05, y_start - i * y_step, line, ha='left', va='top',
            transform=ax_bottom_right.transAxes,
            fontsize=text_fontsize + 1
        )

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    df_1 = pd.read_csv("/Users/avencast/PycharmProjects/EveNet/downstreams/Quantum/pretrain-ema-ds1p0.csv")
    df_2 = pd.read_csv("/Users/avencast/PycharmProjects/EveNet/downstreams/Quantum/scratch-ema-ds1p0-lr_half.csv")
    df_p = pd.read_csv("/Users/avencast/PycharmProjects/EveNet/downstreams/Quantum/paper.csv")

    tag_1 = "pretrain"
    tag_2 = "scratch"
    tag_3 = "EPJC (2022) 82:285"

    for df, tag in [(df_1, tag_1), (df_2, tag_2), (df_p, tag_3)]:
        df.rename(columns={
            'value': f'{tag}_value',
            'uncertainty_up': f'{tag}_up',
            'uncertainty_down': f'{tag}_down',
        }, inplace=True)

    df = pd.merge(df_1, df_2, on=['name'], how='inner')
    # df = pd.merge(df, df_p, on=['name'], how='outer')

    plot_poi(
        df=df,
        model_cols=['pretrain', 'scratch',
                    # 'EPJC (2022) 82:285'
                    ],
        channel_col='name',
        model_configs={
            'pretrain': {'label': None, 'color': '#1f77b4', 'special': True},
            'scratch': {'label': None, 'color': '#ff7f0e', 'special': True},
            # 'EPJC (2022) 82:285': {'label': None, 'color': '#ffff0e', 'special': True},
        },
        channels_configs={
            'D': {'color': '#1f77b4', 'label': r'$D - 1 $', 'special': True},
            'C_kk': {'color': '#E6A23C', 'label': r'$C_{kk}$'},
            'C_rr': {'color': '#CA645F', 'label': r'$C_{rr}$'},
            'C_nn': {'color': '#E6A23C', 'label': r'$C_{nn}$'},
        },
        method_configs={
            'pretrain': {'color': '#156434', 'label': r'EveNet Pretrain (41M)', 'special': None},
            'scratch': {'color': '#32037D', 'label': r'EveNet Scratch (41M)', 'special': None},
            # 'EPJC (2022) 82:285': {'color': '#FF0000', 'label': r'EPJC (2022) 82:285', 'special': None},
        },
        text_precision=2,
        figsize=(8,6),
        text_fontsize=13,
        save_path="poi.pdf",
    )
