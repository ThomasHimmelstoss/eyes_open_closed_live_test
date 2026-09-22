"""
lsl_stream.py - connection to the configured LSL stream + applying a sliding window buffer.
"""

from collections import deque

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop

from config import STEP_SECONDS_LIVE, STREAM_NAME, WINDOW_SECONDS, EXPECTED_CHANNEL_COUNT

RESOLVE_TIMEOUT_S = 5.0


def connect_inlet(stream_name=STREAM_NAME, timeout=RESOLVE_TIMEOUT_S, max_attempts=8):
    for attempt in range(1, max_attempts + 1):
        print(f"[lsl] Searching for stream '{stream_name}' (Attempt {attempt}/{max_attempts}) ...")
        results = resolve_byprop("name", stream_name, timeout=timeout)
        if results:
            inlet = StreamInlet(results[0])
            
            actual_count = inlet.info().channel_count()
            if actual_count != EXPECTED_CHANNEL_COUNT:
                raise RuntimeError(
                    f"Stream '{stream_name}' shows {actual_count} channels, "
                    f"expected are {EXPECTED_CHANNEL_COUNT} (check config.py "
                    f"CHANNEL_NAMES). Checking simulink-Modell and config.py  "
                    f"to avoid working with wrong channels."
                )

            print(f"[lsl] Connected to '{stream_name}'")
            return inlet
        
    raise RuntimeError(
        f"Stream '{stream_name}' was not found after {max_attempts}."
    )


def window_generator(inlet, window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS_LIVE):
    """Generator: gives a (samples, timestamps)-tupel for each step for the current
    sliding window - once enough data is available.

    samples: np.ndarray, shape (n_samples, n_channels)
    timestamps: np.ndarray, shape (n_samples,), LSL-derived time (local_clock())
    """

    buffer = deque()
    last_yield_time = 0.0

    while True:
        sample, ts = inlet.pull_sample(timeout=0.0)
        while sample is not None:
            buffer.append((ts, sample))
            sample, ts = inlet.pull_sample(timeout=0.0)

        now = local_clock()
        while buffer and now - buffer[0][0] > window_seconds:
            buffer.popleft()

        enough_data = buffer and (buffer[-1][0] - buffer[0][0]) >= window_seconds * 0.995
        if enough_data and now - last_yield_time >= step_seconds:
            timestamps = np.array([t for t, _ in buffer])   # lsl local_clock(), for windowing
            samples = np.array([s for _, s in buffer])      # lsl stream data, each entry in "s" is one single FLAT sample vector 
                                                            # that is being sent over from simulink by LSL (opposed to a column vector 
                                                            # within Simulink -> e.g. n_samples (rows), n_LSL_channels (columns, can 
                                                            # be 8 or 28, based on what is selected in config.py))
                                                            # so: axis=0 => time, axis=1 => channels  --> e.g. welch expects this format
            last_yield_time = now
            yield samples, timestamps