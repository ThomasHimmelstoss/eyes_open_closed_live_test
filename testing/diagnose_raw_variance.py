"""
diagnose_raw_variance.py - schaut sich rohe, unverarbeitete LSL-Samples an,
komplett ohne windowing/filtering/feature-Berechnung, um auszuschliessen,
dass diese Schritte selbst eine kuenstliche Periodizitaet erzeugen.
"""

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from config import ALPHA_CHANNEL_INDICES

results = resolve_byprop("name", "EEG_measurement_data_stream", timeout=5.0)
inlet = StreamInlet(results[0])

raw_samples = []
for _ in range(2500):  # 10s bei 250Hz, direkt per pull_sample, kein Buffering-Umweg
    sample, ts = inlet.pull_sample()
    raw_samples.append(sample)

raw_samples = np.array(raw_samples)[:, ALPHA_CHANNEL_INDICES]
print("Std ueber 10s am Stueck:", raw_samples.std(axis=0))

# Jetzt in 2 Haelften teilen und getrennt pruefen - identische Std in
# BEIDEN Haelften waere verdaechtig, unterschiedliche waere normal
half = len(raw_samples) // 2
print("Std erste Haelfte:", raw_samples[:half].std(axis=0))
print("Std zweite Haelfte:", raw_samples[half:].std(axis=0))