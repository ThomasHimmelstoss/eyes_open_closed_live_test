"""
calibration.py

baseline recording ahead of live test - one block for each condition (eyes closed/ open)
used to acquire and compensate individual and hardware-dependent alpha power differences 
compared to the GIPSA training dataset/ model.

important: baseline is pooled over both condition blocks (not per condition) - this replicates
the approach taken in pretrain_gipsa.py. So labels are not used for fitting/ fine-tuning of a classificator.
Only used to record a clean baseline and for a quick sanity-check to see if the effect is present 
for this individual person and setup
"""

import os
from datetime import datetime
import time

import numpy as np
from joblib import dump

from config import (ALPHA_CHANNEL_INDICES, RECORDINGS_DIR, STEP_SECONDS_LIVE,
                     SUBJECT_ID, WINDOW_SECONDS, SAMPLING_RATE_HZ, pad_or_discard_window)
from lsl_stream import connect_inlet, window_generator
from relative_alpha_power_features import relative_alpha_power


# Length of a calibration log
BLOCK_SECONDS = 25.0

# First/last seconds of recording are removed as they do not fully represent the persons current state
EDGE_BUFFER_SECONDS = 2.0

# Amount of samples per window based on windows sec + fs
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE_HZ)  # here: 1000

CALIBRATION_OUTPUT_PATH = "baseline_calibration.joblib"


def record_block(inlet, label, block_seconds=BLOCK_SECONDS):
    print(f"\n[calibration] Next block: '{label}' for {block_seconds:.0f}s.")
    input("[calibration] Press Enter to start ...")
    print("Calibration running ... ")

    features = []
    raw_windows = []   # raw (non-z-scored) EEG for PSD analysis later on
    generator = window_generator(inlet, window_seconds=WINDOW_SECONDS,
                                  step_seconds=STEP_SECONDS_LIVE)
    start_time = time.monotonic()

    for samples, _ in generator:
        elapsed = time.monotonic() - start_time
        if elapsed >= block_seconds:
            break
        if elapsed < EDGE_BUFFER_SECONDS or elapsed > block_seconds - EDGE_BUFFER_SECONDS:
            continue

        eeg_window = samples[:, ALPHA_CHANNEL_INDICES]
        features.append(relative_alpha_power(eeg_window))

        adjusted_window = pad_or_discard_window(eeg_window, WINDOW_SAMPLES)
        if adjusted_window is not None:
            raw_windows.append(adjusted_window)
        else:
            print(f"[calibration]  WARNING: Window with {len(eeg_window)}/{WINDOW_SAMPLES} "
                  f"Sample discarded (too many missing for padding).")

    print(f"[calibration]  '{label}': {len(features)} windows collected.")
    return np.array(features), raw_windows


def compute_baseline_stats(closed_features, open_features):
    """Pooled over both conditions."""
    pooled = np.vstack([closed_features, open_features])
    mean = pooled.mean(axis=0)
    std = np.maximum(pooled.std(axis=0), 1e-3)
    return {"mean": mean, "std": std}


def print_sanity_check(closed_features, open_features):
    """Just a quick sanity check to see if the data lays within a reasonable range and if effect is present."""
    closed_mean = closed_features.mean(axis=0)
    open_mean = open_features.mean(axis=0)
    diff = closed_mean - open_mean
    print("\n[calibration] Sanity-Check (should be > 0, for each channel):")
    print(f"  Closed-mean: {np.round(closed_mean, 3)}")
    print(f"  Open-mean:   {np.round(open_mean, 3)}")
    print(f"  Difference:   {np.round(diff, 3)}")
    if np.any(diff <= 0):
        print("[calibration] Warning: at least one channel shows no or negative effect"
        " - check electrode contact/ setup before proceeding.")


def main():
    session_dir = os.path.join(RECORDINGS_DIR, SUBJECT_ID, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    inlet = connect_inlet()

    closed_features, closed_raw_windows = record_block(inlet, "Close your eyes")
    open_features, open_raw_windows = record_block(inlet, "Open your eyes")

    print_sanity_check(closed_features, open_features)

    np.savez(
        os.path.join(session_dir, "calibration_raw.npz"),
        closed_features=closed_features,
        open_features=open_features,
        closed_raw_windows=np.array(closed_raw_windows),  # NEU: shape (n_windows, n_samples, 4)
        open_raw_windows=np.array(open_raw_windows),        # NEU
        channel_names=["P3", "P4", "O1", "O2"],
        sampling_rate_hz=SAMPLING_RATE_HZ,   # NEU: fuer die spaetere Welch-Berechnung
    )

    stats = compute_baseline_stats(closed_features, open_features)
    dump(stats, os.path.join(session_dir, "baseline_calibration.joblib"))
    print(f"\n[calibration] Session saved to: {session_dir}")


if __name__ == "__main__":
    main()