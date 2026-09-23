"""
pretrain_gipsa.py - single-run offline script: loads GIPSA "EEG Alpha
Waves" dataset (Cattan, Rodrigues & Congedo, 2018) via MOABB, resamples
to config.SAMPLING_RATE_HZ, applies a 50 Hz notch + 1-25 Hz bandpass
filter (to approximate what OSCAR LIVE's artifact-triggered processing
does to the live g.tec signal - see chat), extracts the same alpha-power
feature as relative_alpha_power_features.py, trains a linear classifier,
and saves it for live_classify.py.

Execute once, not part of the live pipeline.
"""

import mne
import numpy as np
from joblib import dump
from moabb.datasets import Rodrigues2017
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from scipy.signal import ellip, iirnotch, sosfilt, tf2sos

from config import ALPHA_CHANNEL_NAMES, SAMPLING_RATE_HZ, STEP_SECONDS_TRAIN, WINDOW_SECONDS
from relative_alpha_power_features import relative_alpha_power

MODEL_OUTPUT_PATH = "alpha_classifier.joblib"
TRAINING_FEATURES_OUTPUT_PATH = "gipsa_training_features.npz"

# Preprocessing applied to GIPSA's raw (unfiltered) data before feature
# extraction, to bring it structurally closer to the live g.tec signal:
#   - notch: 50 Hz line noise, same as the live Simulink Notch block
#   - bandpass: 1-25 Hz - approximates OSCAR LIVE's 0-25 Hz frequency
#     range on the upper end, and excludes DC drift on the lower end.
#     Assumes OSCAR's artifact-triggered correction leaves calm,
#     artifact-free segments close to untouched (see chat) - the live
#     signal itself stays on the OSCAR-cleaned path, with an additional
#     1 Hz high-pass added in Simulink after the OSCAR block to align the
#     lower bound on both sides.
APPLY_PREPROCESSING = ("notch", "bandpass")

# Event coding per the MOABB source for Rodrigues2017. Verified once
# against a real Raw object (see check_gipsa_annotations.py) - annotation
# labels are exactly {'closed', 'open'}.
EVENT_ID = {"closed": 1, "open": 2}
BLOCK_SECONDS = 10.0  # block length in the original experiment protocol


def load_raws():
    """Loads all subjects as a list of (raw, subject_id) tuples.

    get_data() returns a nested dict {subject: {session: {run: raw}}}.
    We flatten it here instead of assuming exactly 1 session/1 run per
    subject, so an unexpected structure surfaces immediately instead of
    silently dropping data.

    Note: GIPSA subject 7 does not exist in MOABB's subject_list (invalid
    ID, confirmed via dataset.subject_list) - load_raws() therefore
    yields 19 subjects, not 20. This is a property of the dataset as
    distributed via MOABB, not a bug in this loader.
    """
    dataset = Rodrigues2017()
    data = dataset.get_data()

    raws = []
    for subject_id, sessions in data.items():
        for session_id, runs in sessions.items():
            for run_id, raw in runs.items():
                raws.append((raw, subject_id))
    return raws


def apply_filters(data, fs, filter_names):
    """Applies filters matching the exact Simulink Notch design, plus a
    single elliptic bandpass (1-25 Hz) using the SAME filter
    characteristics as the real "1 Hz Highpass Filter" Simulink block
    (order 4, 0.1 dB ripple, 60 dB stopband attenuation) for the lower
    edge. The upper 25 Hz edge has no real Simulink block to copy from
    (GIPSA never passes through OSCAR) - reusing the same elliptic design
    for both edges is a consistent, if unverified, approximation of
    OSCAR's 0-25 Hz output range.
    """
    if "notch" in filter_names:
        b, a = iirnotch(w0=50.0, Q=25.0, fs=fs) # usually FIR filters and filtfilt is prefered for offline filtering - here only different to replicate online simulink filters
        sos = tf2sos(b, a)
        data = sosfilt(sos, data, axis=0)

    if "bandpass" in filter_names:
        sos = ellip(N=4, rp=0.1, rs=60, Wn=[1.0, 25.0], btype="bandpass", fs=fs, output="sos")  # same here (as for notch)
        data = sosfilt(sos, data, axis=0)

    return data


def extract_windows_and_labels(raw):
    """Cuts all 4s windows (step STEP_SECONDS_TRAIN) out of a continuous
    Raw object that lie FULLY within a single 10s block - no window
    crosses a block/condition boundary.

    Returned features are already z-normalized PER SUBJECT (mean/std
    over ALL windows of this subject, both conditions pooled) - the exact
    same transform that calibration.py/live_classify.py later apply for
    a new person. This is the shared feature space the model is trained
    in.

    Returns: (features, labels)
    features: shape (n_windows, n_channels)
    labels:   shape (n_windows,), 1 = eyes closed, 0 = eyes open
    """
    raw = raw.copy().pick(ALPHA_CHANNEL_NAMES)  # P3, P4, O1, O2 only
    raw.resample(SAMPLING_RATE_HZ, method="polyphase")  # 512 Hz -> 250 Hz

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID)
    fs = raw.info["sfreq"]
    block_samples = int(BLOCK_SECONDS * fs)
    window_samples = int(WINDOW_SECONDS * fs)
    step_samples = int(STEP_SECONDS_TRAIN * fs)

    data = raw.get_data().T * 1e6  # MNE stores Volts, convert to microvolts
    data = apply_filters(data, fs, APPLY_PREPROCESSING)  # matches the exact Simulink filter designs

    features, labels = [], []
    for onset_sample, _, label_code in events:
        block_end = onset_sample + block_samples
        label = 1 if label_code == EVENT_ID["closed"] else 0

        start = onset_sample
        while start + window_samples <= block_end:
            window = data[start:start + window_samples]
            features.append(relative_alpha_power(window, fs=fs))
            labels.append(label)
            start += step_samples

    features = np.array(features)
    labels = np.array(labels)

    subject_mean = features.mean(axis=0)
    subject_std = np.maximum(features.std(axis=0), 1e-3)
    normalized_features = (features - subject_mean) / subject_std

    return normalized_features, labels


def main():
    raws = load_raws()
    print(f"[pretrain] {len(raws)} recordings loaded from GIPSA.")

    all_features, all_labels, all_groups = [], [], []
    for raw, subject_id in raws:
        features, labels = extract_windows_and_labels(raw)  # already per-subject normalized
        all_features.append(features)
        all_labels.append(labels)
        all_groups.append(np.full(len(labels), subject_id))

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    print(f"[pretrain] {X.shape[0]} windows total, {X.shape[1]} features/window.")

    np.savez(
        TRAINING_FEATURES_OUTPUT_PATH,
        X=X, y=y, groups=groups,
        channel_names=ALPHA_CHANNEL_NAMES,
    )
    print(f"[pretrain] Training features saved to: {TRAINING_FEATURES_OUTPUT_PATH}")

    # No StandardScaler here: X is already per-subject z-normalized. An
    # additional global scaler would only slightly shift the values, but
    # more importantly it would reintroduce the exact problem we already
    # fixed - the live value is likewise already individually z-scored,
    # not raw.
    model = LogisticRegression()

    # Subject-wise GroupKFold instead of standard KFold: windows from the
    # SAME subject must never end up in both train and test at once, or
    # CV would systematically overestimate accuracy (the real question is
    # whether this generalizes to a NEW person, not a already-seen one).
    scores = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(n_splits=5))
    print(f"[pretrain] Cross-subject accuracy: {scores.mean():.3f} "
          f"(+/- {scores.std():.3f})")

    model.fit(X, y)
    dump(model, MODEL_OUTPUT_PATH)
    print(f"[pretrain] Model saved: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()