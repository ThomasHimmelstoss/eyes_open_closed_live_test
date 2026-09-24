"""
testing/diagnose_connect_transient.py - prueft, ob unmittelbar nach
connect_inlet() ein ungewoehnliches Timing-Muster auftritt (z.B. Backlog-
Nachlieferung), das record_block()'s ERSTES Fenster beeinflussen koennte.
"""

from pylsl import local_clock
from lsl_stream import connect_inlet

inlet = connect_inlet()

print("Erste 30 Sample-Ankunftszeiten nach Verbindungsaufbau:")
previous_ts = None
for i in range(30):
    sample, ts = inlet.pull_sample()
    if previous_ts is not None:
        print(f"  Sample {i}: Delta zum Vorgaenger = {(ts - previous_ts)*1000:.2f} ms")
    previous_ts = ts