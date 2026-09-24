"""
diagnose_window_overlap.py - prueft, ob aufeinanderfolgende Fenster aus
window_generator() tatsaechlich NEUE Daten enthalten, oder ob sich
Abschnitte verdaechtig oft exakt wiederholen.
"""

import numpy as np

from config import ALPHA_CHANNEL_INDICES, STEP_SECONDS_LIVE, WINDOW_SECONDS
from lsl_stream import connect_inlet, window_generator

inlet = connect_inlet()

previous_window = None
for i, (samples, _) in enumerate(window_generator(inlet, window_seconds=WINDOW_SECONDS,
                                                    step_seconds=STEP_SECONDS_LIVE)):
    eeg_window = samples[:, ALPHA_CHANNEL_INDICES]

    if previous_window is not None:
        # Der NEUE Teil eines Fensters (die letzten STEP_SECONDS*fs Samples)
        # sollte sich vom letzten Teil des VORHERIGEN Fensters unterscheiden,
        # wenn wirklich neue Daten reinkommen.
        overlap_matches = np.array_equal(eeg_window[-50:], previous_window[-50:])
        print(f"Fenster {i}: letzte 50 Samples identisch zum Vorgaenger? {overlap_matches}")

    previous_window = eeg_window
    if i >= 10:
        break