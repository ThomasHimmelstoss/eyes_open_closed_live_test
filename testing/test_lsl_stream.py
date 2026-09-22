"""
test_lsl_stream.py - einmaliges, temporaeres Testskript fuer Schritt 1.
Nicht Teil der eigentlichen Pipeline, nur zum Debuggen.
"""

from lsl_stream import connect_inlet, window_generator
from config import WINDOW_SECONDS, STEP_SECONDS_LIVE

inlet = connect_inlet()

print(f"[test] Stream verbunden. Erwarte Fenster von {WINDOW_SECONDS}s alle "
      f"{STEP_SECONDS_LIVE}s ...\n")

for i, (samples, timestamps) in enumerate(
        window_generator(inlet, window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS_LIVE)):
    print(f"--- Fenster {i} ---")
    print(f"  shape: {samples.shape}")          # erwartet: (~1000, 9) bei 250 Hz * 4s
    print(f"  erste Zeile (roh):  {samples[0]}")
    print(f"  Zeitspanne timestamps: {timestamps[-1] - timestamps[0]:.3f}s")

    if i >= 15:  # nach 5 Fenstern abbrechen, reicht zum Pruefen
        break