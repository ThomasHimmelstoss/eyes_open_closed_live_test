"""
analyze_psd_calibration.py - PSD comparison (closed vs. open) from your
own calibration recordings: a single session, or the grand average
across ALL calibration sessions found under RECORDINGS_DIR (i.e. across
all your subjects/sessions so far).

Usage:
  python analyze_psd_calibration.py session recordings/subject_01/2026-09-18_143022
  python analyze_psd_calibration.py all
"""

import glob
import sys

import numpy as np

from config import ALPHA_CHANNEL_NAMES, RECORDINGS_DIR
from psd_analysis import average_across_groups, compute_condition_psd, plot_psd_comparison


def load_session_windows(session_dir):
    data = np.load(f"{session_dir}/calibration_raw.npz", allow_pickle=True)
    return data["closed_raw_windows"], data["open_raw_windows"], int(data["sampling_rate_hz"])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "session":
        session_dir = sys.argv[2]
        closed_windows, open_windows, fs = load_session_windows(session_dir)

        freqs, psd_closed = compute_condition_psd(closed_windows, fs)
        _, psd_open = compute_condition_psd(open_windows, fs)

        plot_psd_comparison(freqs, psd_closed, psd_open, ALPHA_CHANNEL_NAMES,
                             f"Calibration session {session_dir} - PSD closed vs. open",
                             f"{session_dir}/psd_comparison.png")

    elif mode == "all":
        session_paths = sorted(glob.glob(f"{RECORDINGS_DIR}/*/*/calibration_raw.npz"))
        if not session_paths:
            print(f"No calibration sessions found under {RECORDINGS_DIR}/")
            sys.exit(1)

        per_session_closed, per_session_open = [], []
        for path in session_paths:
            session_dir = path.rsplit("/", 1)[0]
            closed_windows, open_windows, fs = load_session_windows(session_dir)
            per_session_closed.append(compute_condition_psd(closed_windows, fs))
            per_session_open.append(compute_condition_psd(open_windows, fs))
            print(f"  processed {session_dir}")

        freqs, psd_closed = average_across_groups(per_session_closed)
        _, psd_open = average_across_groups(per_session_open)

        plot_psd_comparison(freqs, psd_closed, psd_open, ALPHA_CHANNEL_NAMES,
                             f"Grand average across {len(session_paths)} calibration sessions - "
                             f"PSD closed vs. open",
                             "calibration_grand_average_psd.png")
    else:
        print(f"Unknown mode '{mode}'. Use 'session <dir>' or 'all'.")
        sys.exit(1)


if __name__ == "__main__":
    main()