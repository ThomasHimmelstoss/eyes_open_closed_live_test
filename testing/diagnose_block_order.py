"""
testing/diagnose_block_order.py
"""

import os
from calibration import connect_inlet, record_block
from lsl_stream import discard_initial_windows
import numpy as np
from psd_analysis import compute_condition_psd, plot_psd_comparison

os.makedirs("testing/block_order_check", exist_ok=True)

inlet = connect_inlet()
discard_initial_windows(inlet)

features_a, raw_a = record_block(inlet, "Block A (neutral, keine Augen-Instruktion)")
features_b, raw_b = record_block(inlet, "Block B (neutral, keine Augen-Instruktion)")

np.savez(
    "testing/block_order_check/calibration_raw.npz",
    closed_raw_windows=np.array(raw_a),
    open_raw_windows=np.array(raw_b),
    channel_names=["P3", "P4", "O1", "O2"],
    sampling_rate_hz=250,
)

# Konsistent denselben Pfad UND dieselben Schluessel wie beim Speichern
data = np.load("testing/block_order_check/calibration_raw.npz", allow_pickle=True)
raw_a = data["closed_raw_windows"]
raw_b = data["open_raw_windows"]

freqs, psd_a = compute_condition_psd(raw_a, fs=250)
_, psd_b = compute_condition_psd(raw_b, fs=250)

plot_psd_comparison(freqs, psd_a, psd_b, ["P3", "P4", "O1", "O2"],
                     "Block A vs. Block B (neutral, keine Augen-Instruktion)",
                     "testing/block_order_check_psd.png")