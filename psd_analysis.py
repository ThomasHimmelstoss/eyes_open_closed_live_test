"""
psd_analysis.py - shared PSD computation and plotting, reused by both
analyze_psd_gipsa.py (training set) and analyze_psd_calibration.py (your
own recordings). Mirrors the spectral analysis from the original GIPSA
example script, generalized to all 4 alpha channels and to averaging
across many windows/subjects instead of a single subject/channel.
"""

import numpy as np
from scipy.signal import welch


def compute_condition_psd(windows, fs, freq_max=40.0):
    """windows: list/array of (n_samples, n_channels) raw EEG windows, all
    from the SAME condition (all 'closed' or all 'open').

    Returns: freqs (n_freqs,), psd_mean (n_freqs, n_channels) - averaged
    across all given windows, restricted to [0, freq_max] Hz.
    """
    windows = np.asarray(windows)  # (n_windows, n_samples, n_channels)
    freqs, psd = welch(windows, fs=fs, axis=1)  # psd: (n_windows, n_freqs, n_channels)
    psd_mean = psd.mean(axis=0)  # average over windows -> (n_freqs, n_channels)

    freq_mask = freqs <= freq_max
    return freqs[freq_mask], psd_mean[freq_mask]


def average_across_groups(freqs_and_psds):
    """Averages several (freqs, psd) results with EQUAL weight per group,
    regardless of how many windows each group contributed.

    Used for grand averages (e.g. across GIPSA subjects, or across your
    own calibration sessions) - this is standard EEG grand-averaging
    practice: a subject/session with more windows should not dominate
    the average just by having contributed more data.
    """
    freqs = freqs_and_psds[0][0]  # all groups share the same freq axis (same fs, same welch settings)
    psd_stack = np.stack([psd for _, psd in freqs_and_psds], axis=0)  # (n_groups, n_freqs, n_channels)
    return freqs, psd_stack.mean(axis=0)


def plot_psd_comparison(freqs, psd_closed, psd_open, channel_names, title, output_path, log_scale=False):
    import matplotlib.pyplot as plt

    n_channels = len(channel_names)
    fig, axes = plt.subplots(1, n_channels, figsize=(4.5 * n_channels, 4.5), sharey=False)
    if n_channels == 1:
        axes = [axes]

    plot_fn_name = "semilogy" if log_scale else "plot"

    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        plot_fn = getattr(ax, plot_fn_name)
        plot_fn(freqs, psd_closed[:, i], color="k", lw=2.5, label="closed")
        plot_fn(freqs, psd_open[:, i], color="r", lw=2.5, label="open")
        ax.axvspan(8, 12, color="tab:blue", alpha=0.08)
        ax.set_title(name)
        ax.set_xlabel("frequency (Hz)")
        if i == 0:
            ax.set_ylabel("power spectral density" + (" (log scale)" if log_scale else ""))
        ax.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")