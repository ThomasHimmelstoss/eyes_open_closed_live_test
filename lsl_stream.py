"""
lsl_stream.py - connection to the configured LSL stream + applying a sliding window buffer.

If sender (Simulink) and receiver (this script) run on different physical machines,
both have their own, unsychronized local clocks (steady clock). LSL-timestamps of the sender
are not directly comparable to local_clock of the receiver -> inlet.time_correction() measures
the offset between both clocks and in this script we add the offset to every raw timestamp
-> translates it into our own timebase before using it for windowing. 
"""

from collections import deque

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop

from config import STEP_SECONDS_LIVE, STREAM_NAME, WINDOW_SECONDS

RESOLVE_TIMEOUT_S = 5.0
OFFSET_REFRESH_INTERVAL_S = 30.0


def connect_inlet(stream_name=STREAM_NAME, timeout=RESOLVE_TIMEOUT_S, max_attempts=8):
    for attempt in range(1, max_attempts + 1):
        print(f"[lsl] Suche Stream '{stream_name}' (Versuch {attempt}/{max_attempts}) ...")
        results = resolve_byprop("name", stream_name, timeout=timeout)
        if results:
            inlet = StreamInlet(results[0])
            print(f"[lsl] verbunden mit '{stream_name}'")
            return inlet
    raise RuntimeError(f"Stream '{stream_name}' nach {max_attempts} Versuchen nicht gefunden.")


def window_generator(inlet, window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS_LIVE):
    buffer = deque()
    last_yield_time = 0.0

    clock_offset = 0.0
    last_offset_refresh = -OFFSET_REFRESH_INTERVAL_S

    while True:
        now = local_clock()
        if now - last_offset_refresh >= OFFSET_REFRESH_INTERVAL_S:
            try:
                clock_offset = inlet.time_correction(timeout=1.0)
            except TimeoutError:
                pass  # alten Offset behalten statt auf 0 zurueckzufallen
            last_offset_refresh = now

        sample, ts = inlet.pull_sample(timeout=0.0)
        while sample is not None:
            corrected_ts = ts + clock_offset  # in unsere eigene Zeitbasis uebersetzt
            buffer.append((corrected_ts, sample))
            sample, ts = inlet.pull_sample(timeout=0.0)

        now = local_clock()
        while buffer and now - buffer[0][0] > window_seconds:
            buffer.popleft()

        enough_data = buffer and (buffer[-1][0] - buffer[0][0]) >= window_seconds * 0.995
        if enough_data and now - last_yield_time >= step_seconds:
            timestamps = np.array([t for t, _ in buffer])
            samples = np.array([s for _, s in buffer])
            last_yield_time = now
            yield samples, timestamps
            
            
def discard_initial_windows(inlet, n_windows=4, window_seconds=WINDOW_SECONDS,
                              step_seconds=STEP_SECONDS_LIVE):
    """Consumes and discards the first n_windows from window_generator()
    right after connecting, as a safety margin against any connection-
    startup artifact (backlog catch-up, OS scheduling on first connect,
    etc. - exact cause not conclusively identified, see chat). Cheap
    (a few seconds) and removes the risk regardless of root cause.
    """
    generator = window_generator(inlet, window_seconds=window_seconds, step_seconds=step_seconds)
    for _ in range(n_windows):
        next(generator)