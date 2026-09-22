"""
diagnose_rate_over_time.py - misst die Ankunftsrate in 5s-Intervallen ueber
laengere Zeit, mit Retry beim Verbindungsaufbau (Multicast-Discovery ist
bei uns nachweislich flakey, siehe Chat).
"""

import time

from pylsl import StreamInlet, resolve_byprop

STREAM_NAME = 'EEG_measurement_data_stream'
MAX_CONNECT_ATTEMPTS = 8
MEASURE_SECONDS = 90.0
REPORT_INTERVAL = 5.0


def connect(stream_name, max_attempts=MAX_CONNECT_ATTEMPTS):
    for attempt in range(1, max_attempts + 1):
        print(f"[diagnose] Suche '{stream_name}' (Versuch {attempt}/{max_attempts}) ...")
        results = resolve_byprop("name", stream_name, timeout=5.0)
        if results:
            print(f"[diagnose] verbunden mit '{stream_name}'")
            return StreamInlet(results[0])
    raise RuntimeError(f"'{stream_name}' nach {max_attempts} Versuchen nicht gefunden.")


inlet = connect(STREAM_NAME)

start = time.time()
last_report = start
count_since_last = 0

while time.time() - start < MEASURE_SECONDS:
    sample, ts = inlet.pull_sample(timeout=0.1)
    if sample is not None:
        count_since_last += 1

    now = time.time()
    if now - last_report >= REPORT_INTERVAL:
        rate = count_since_last / (now - last_report)
        elapsed_total = now - start
        print(f"[diagnose] t={elapsed_total:5.1f}s  letzte {REPORT_INTERVAL:.0f}s: {rate:.1f} Hz")
        count_since_last = 0
        last_report = now