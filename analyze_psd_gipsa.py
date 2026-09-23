"""
analyze_psd_gipsa.py - PSD comparison (closed vs. open) for the GIPSA
training set: a single subject, or the grand average across all subjects.
Uses the SAME filtering as pretrain_gipsa.py (imported directly, not
duplicated), so this plot always reflects exactly what the model was
actually trained on.

Usage:
  python analyze_psd_gipsa.py subject 3
  python analyze_psd_gipsa.py all
"""

import sys

import mne
import numpy as np

from config import ALPHA_CHANNEL_NAMES, SAMPLING_RATE_HZ
from psd_analysis import average_across_groups, compute_condition_psd, plot_psd_comparison
from pretrain_gipsa import APPLY_PREPROCESSING, BLOCK_SECONDS, EVENT_ID, apply_filters, load_raws


def extract_raw_windows_for_subject(raw, window_seconds=4.0, step_seconds=2.0):
    raw = raw.copy().pick(ALPHA_CHANNEL_NAMES)
    raw.resample(SAMPLING_RATE_HZ, method="polyphase")

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID)
    fs = raw.info["sfreq"]
    block_samples = int(BLOCK_SECONDS * fs)
    window_samples = int(window_seconds * fs)
    step_samples = int(step_seconds * fs)

    data = raw.get_data().T * 1e6  # Volts -> microvolts
    data = apply_filters(data, fs, APPLY_PREPROCESSING)  # same filtering as training, imported above

    closed_windows, open_windows = [], []
    for onset_sample, _, label_code in events:
        block_end = onset_sample + block_samples
        target_list = closed_windows if label_code == EVENT_ID["closed"] else open_windows

        start = onset_sample
        while start + window_samples <= block_end:
            target_list.append(data[start:start + window_samples])
            start += step_samples

    return closed_windows, open_windows, fs


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    filter_label = "+".join(APPLY_PREPROCESSING)  # e.g. "notch+bandpass" - reflects reality, not a flag
    raws = load_raws()

    if mode == "subject":
        subject_id = int(sys.argv[2])
        raw = next(r for r, sid in raws if sid == subject_id)
        closed_windows, open_windows, fs = extract_raw_windows_for_subject(raw)

        freqs, psd_closed = compute_condition_psd(closed_windows, fs)
        _, psd_open = compute_condition_psd(open_windows, fs)

        plot_psd_comparison(freqs, psd_closed, psd_open, ALPHA_CHANNEL_NAMES,
                             f"GIPSA subject {subject_id} ({filter_label}) - PSD closed vs. open",
                             f"gipsa_subject_{subject_id}_psd.png")

    elif mode == "all":
        per_subject_closed, per_subject_open = [], []
        for raw, subject_id in raws:
            closed_windows, open_windows, fs = extract_raw_windows_for_subject(raw)
            per_subject_closed.append(compute_condition_psd(closed_windows, fs))
            per_subject_open.append(compute_condition_psd(open_windows, fs))
            print(f"  processed subject {subject_id}")

        freqs, psd_closed = average_across_groups(per_subject_closed)
        _, psd_open = average_across_groups(per_subject_open)

        plot_psd_comparison(freqs, psd_closed, psd_open, ALPHA_CHANNEL_NAMES,
                             f"GIPSA grand average ({filter_label}, n={len(raws)} subjects) - "
                             f"PSD closed vs. open",
                             "gipsa_grand_average_psd.png")
    else:
        print(f"Unknown mode '{mode}'. Use 'subject <id>' or 'all'.")
        sys.exit(1)


if __name__ == "__main__":
    main()