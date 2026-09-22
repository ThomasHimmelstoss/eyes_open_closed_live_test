"""
test_multi_device.py - mechanical sanity check for multi_device_lsl.py,
using the same EEG stream twice as a stand-in for "two devices" (both on
this machine, so offsets should stay ~0).
"""

import time

from multi_device_lsl import connect_devices, refresh_all_offsets

from pylsl import resolve_streams

streams = resolve_streams(wait_time=5.0)
for info in streams:
    print(f"Name: '{info.name()}'  Typ: {info.type()}  "
          f"Kanaele: {info.channel_count()}  Rate: {info.nominal_srate()} Hz  "
          f"Quelle (Hostname): {info.hostname()}")
    

tracked = connect_devices({
    "eeg_a": "EEG_measurement_data_stream",
    "eeg_b": "EEG_measurement_data_stream",
})

for _ in range(5):
    refresh_all_offsets(tracked)
    for label, t in tracked.items():
        print(f"  {label}: offset={t.offset_seconds:.6f}s")
    time.sleep(1.0)