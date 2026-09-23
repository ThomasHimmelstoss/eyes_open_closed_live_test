"""
config.py - central configuration file for the Python-Realtime-Pipeline.
Just one constant block - always connect to a single LSL-stream.
Settings for:
- Subject/ recording storage
- LSL stream 
- Window size
"""

import numpy as np

# --- Session/subject identification -------------------------------------
SUBJECT_ID = "subject_01"  # change per person before running calibration.py
RECORDINGS_DIR = "recordings"


# --- Catching the LSL stream -----------------------------------
# only activate one at a time

# g.tec 8ch EEG Simulink LSL Stream Data contains:
# LSL Channel Structure:
# 1-8   = OSCAR Filtered EEG (Notch-filtert + OSCAR artifact removal (Frequency Range: 0-25 Hz))
#   9   = OSCAR Artifact Level (logged Artifact Level within the signal)
# 10-17 = Notch+HP Filtered EEG (50 Hz Notch + 0.5 Hz High-Pass filtered ohne OSCAR)
# 18-25 = Raw EEG (unfiltered raw signal)
# 26-28 = Acceleration Data (3D-acceleration data recorded from the EEG headset)
STREAM_NAME = 'EEG_measurement_data_stream'

SAMPLING_RATE_HZ = 250

# --- Available LSL stream channels -----------
# current setup: only the OSCAR related LSL channels are taken from the LSL stream
# remove comments according to project needs

EXPECTED_CHANNEL_COUNT = 28

CHANNEL_GROUPS = {
    "oscar_filtered": {
        0: "FC5", 1: "FC6", 2: "C3", 3: "C4",
        4: "P3", 5: "P4", 6: "O1", 7: "O2",
    },
    "oscar_artifact_level": {
        8: "Artifact Level",
    },
    "notch_hp_filtered": {
        9: "FC5", 10: "FC6", 11: "C3", 12: "C4",
        13: "P3", 14: "P4", 15: "O1", 16: "O2",
    },
    "raw": {
        17: "FC5", 18: "FC6", 19: "C3", 20: "C4",
        21: "P3", 22: "P4", 23: "O1", 24: "O2",
    },
    "acceleration": {
        25: "Acceleration_Data_1", 26: "Acceleration_Data_2", 27: "Acceleration_Data_3",
    },
}


# Select the LSL pipeline channels with the respective EEG data (oscar_filtered, notch_hp_filtered, raw)
ALPHA_SOURCE_GROUP = "oscar_filtered"

# Channels used for the Alpha-Blocking-Classifier (eyes open/closed):
# Occipital + parietal, best SNR for the alpha-blocking effect -
# FC5/FC6/C3/C4 excluded.
ALPHA_CHANNEL_NAMES = ["P3", "P4", "O1", "O2"]

_active_channels = CHANNEL_GROUPS[ALPHA_SOURCE_GROUP]

ALPHA_CHANNEL_INDICES = [
    idx for idx, name in _active_channels.items() if name in ALPHA_CHANNEL_NAMES
]

# Quick check to avoid quietly feeding the model fewer/more features than it was trained on.
assert len(ALPHA_CHANNEL_INDICES) == len(ALPHA_CHANNEL_NAMES), (
    f"Expected {len(ALPHA_CHANNEL_NAMES)} alpha channels in group "
    f"'{ALPHA_SOURCE_GROUP}', found {len(ALPHA_CHANNEL_INDICES)}. "
    f"Check CHANNEL_GROUPS for typos or missing entries."
)

# --- Window-Settings --------------------------------------------------
# 4s window for stable Alpha-Bandpower estimation, 50% Overlap for training, 75% overlap for real-time stream
# using less overlap for training to remove strong statistical dependency between neighboring samples
# to avoid that the model learns these patterns
# more overlap during actual live test to include more data to feed into the decision smoothing
# cauton! dont change WINDOW_SECONDS between training and live test - epochs need to have same overall length.
WINDOW_SECONDS = 4.0
STEP_SECONDS_TRAIN = 2.0   # GIPSA-Pretraining: less overlap = less pseudo-replication 
STEP_SECONDS_LIVE = 1.0    # more frequent updates for decision smoothing

# --- Alpha-Band-Definition ----------------------------------------------
ALPHA_BAND_HZ = (8.0, 12.0)
BROADBAND_HZ = (1.0, 25.0)  # denominator for relative Alpha-Power - 1-25 Hz, bc OSCAR already cuts Fq range accordingly



# Extrapolate missing samples based on window-mean-padding if there are missing samples within a window

MAX_PAD_SAMPLES = 10  # up to 10 missing samples are accepted as a certain range of jitter that can still be reliably 
                     # padded without heavily influencing the actual window raw data 


def pad_or_discard_window(eeg_window, target_length, max_pad_samples=MAX_PAD_SAMPLES):
    """padds or trims a window to exactly target_length, discard if not within reasonable range:
    - too long -> cuts signal
    - missing samples -> max_pad_samples rows -> filled with window mean per channel 
    - if more than max_pad_samples rows missing -> return None (window is discarded, likely problem in the pipeline)
    """
    current_length = len(eeg_window)

    if current_length >= target_length:
        return eeg_window[:target_length]

    missing = target_length - current_length
    if missing > max_pad_samples:
        return None

    channel_means = eeg_window.mean(axis=0)  # (n_channels,)
    padding = np.tile(channel_means, (missing, 1))  # (missing, n_channels)
    return np.vstack([eeg_window, padding])