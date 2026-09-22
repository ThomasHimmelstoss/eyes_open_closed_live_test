"""
multi_device_lsl.py - infrastructure for connecting to and time-aligning
MULTIPLE simultaneous LSL streams from potentially different physical
machines (e.g. EEG from the Simulink PC + eye tracking from a Pupil Labs
Neon laptop + GSR from a Shimmer3 on a third device).

NOT used yet by the current single-stream Alpha-Power pipeline
(lsl_stream.py) - this is scaffolding for future multi-modal projects,
built now while the underlying LSL concepts are fresh (see chat: LSL
time synchronization).

Key idea: each inlet has its OWN clock domain if it originates from a
different physical machine. inlet.time_correction() measures the offset
between that machine's clock and ours via an NTP-like exchange (see LSL
docs). Adding that offset to a raw timestamp maps it into OUR local time
domain, making it comparable to timestamps from any other corrected
stream. On the same machine this offset is genuinely ~0 - see the very
first lsl_recorder.py sketch in this project's history, which already
anticipated exactly this.
"""

from dataclasses import dataclass, field

from pylsl import StreamInlet, local_clock, resolve_byprop

RESOLVE_TIMEOUT_S = 5.0
OFFSET_REFRESH_INTERVAL_S = 30.0  # how often to re-measure each inlet's clock offset


@dataclass
class TrackedInlet:
    """One connected stream plus its most recently measured clock offset.

    offset_seconds: add this to a RAW timestamp from this inlet to map it
    into our local time domain. Starts at 0.0, a safe default for
    same-machine streams (offset is genuinely ~0 there).
    """
    name: str
    inlet: StreamInlet
    offset_seconds: float = 0.0
    last_refresh_time: float = field(default_factory=lambda: -OFFSET_REFRESH_INTERVAL_S)

    def corrected_timestamp(self, raw_timestamp):
        return raw_timestamp + self.offset_seconds

    def refresh_offset_if_due(self, now=None):
        """Re-measures the clock offset if OFFSET_REFRESH_INTERVAL_S has
        passed since the last measurement. Call this once per iteration
        of your main loop - it no-ops most of the time, so calling it
        often is cheap.
        """
        now = now if now is not None else local_clock()
        if now - self.last_refresh_time < OFFSET_REFRESH_INTERVAL_S:
            return

        try:
            self.offset_seconds = self.inlet.time_correction(timeout=1.0)
        except TimeoutError:
            # Keep the previous offset instead of resetting to 0 - a
            # single failed measurement is not evidence the old offset is
            # wrong, and "stale but close" beats "wrong reset to 0".
            pass
        self.last_refresh_time = now


def connect_inlet(stream_name, timeout=RESOLVE_TIMEOUT_S, max_attempts=8):
    """Same retry-based resolve as lsl_stream.connect_inlet() (duplicated,
    not imported) - keeps this module standalone/reusable without a
    dependency on the EEG-specific config.py.
    """
    for attempt in range(1, max_attempts + 1):
        print(f"[multi_device] Searching '{stream_name}' (attempt {attempt}/{max_attempts}) ...")
        results = resolve_byprop("name", stream_name, timeout=timeout)
        if results:
            print(f"[multi_device] connected to '{stream_name}'")
            return StreamInlet(results[0])

    raise RuntimeError(f"Stream '{stream_name}' not found after {max_attempts} attempts.")


def connect_devices(stream_names):
    """stream_names: dict mapping a short device label to its LSL stream
    name, e.g. {"eeg": "EEG_measurement_data_stream", "gaze": "pupil_capture"}

    Returns: dict[label] -> TrackedInlet. Call refresh_all_offsets() on
    the result periodically from your main loop.
    """
    return {
        label: TrackedInlet(name=label, inlet=connect_inlet(stream_name))
        for label, stream_name in stream_names.items()
    }


def refresh_all_offsets(tracked_inlets):
    """Call once per main-loop iteration - refreshes whichever inlets are
    due, no-ops for the rest."""
    now = local_clock()
    for tracked in tracked_inlets.values():
        tracked.refresh_offset_if_due(now)