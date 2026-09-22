"""
pretrain_gipsa.py - single-run offline script: loads GIPSA "EEG Alpha
Waves" dataset (Cattan, Rodrigues & Congedo, 2018) via MOABB, resamples
to config.SAMPLING_RATE_HZ, extracts same Alpha-Power-Features
as relative_alpha_power_features.py and trains a linear classifier for live_classify.py.

Execute once, not part of the Live-Pipeline.
"""

import mne
import numpy as np
from joblib import dump
from moabb.datasets import Rodrigues2017
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

from config import ALPHA_CHANNEL_NAMES, SAMPLING_RATE_HZ, STEP_SECONDS_TRAIN, WINDOW_SECONDS
from relative_alpha_power_features import relative_alpha_power

MODEL_OUTPUT_PATH = "alpha_classifier.joblib"

# Event-Coding according to MOABB-source code for Rodrigues2017. 
EVENT_ID = {"closed": 1, "open": 2}
BLOCK_SECONDS = 10.0  # block length of GISPA recordings


def load_raws():
    """Loads all subjects as a list of (raw, subject_id) tuples.

    get_data() gives nested dict: {subject: {session: {run:
    raw}}} which is then unpacked to get the raw data for each subject.
    """
    dataset = Rodrigues2017()
    data = dataset.get_data()

    raws = []
    for subject_id, sessions in data.items():
        for session_id, runs in sessions.items():
            for run_id, raw in runs.items():
                raws.append((raw, subject_id))
    return raws


def extract_windows_and_labels(raw):
    """Cuts out 4s windows of the raw data (stepwise with STEP_SECONDS_TRAIN), 
    that fully lay in between a 10s block - no windows over the condition/ windows borders.

    The extracted features are then z-score normalized to their own distribution (mean/std over all windows of this subject,
    pooled over both conditions) - identical to calibration.py/live_classify.py later on to create a similar feature space.

    returns: (features, labels)
    features: shape (n_windows, n_channels)
    labels:   shape (n_windows,), 1 = eyes closed, 0 = eyes open
    """
    raw = raw.copy().pick(ALPHA_CHANNEL_NAMES)  # only P3, P4, O1, O2
    raw.resample(SAMPLING_RATE_HZ, method="polyphase")  # 512 Hz -> 250 Hz

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID)
    fs = raw.info["sfreq"]
    block_samples = int(BLOCK_SECONDS * fs)
    window_samples = int(WINDOW_SECONDS * fs)
    step_samples = int(STEP_SECONDS_TRAIN * fs)

    data = raw.get_data().T  # (n_samples, n_channels)

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
    print(f"[pretrain] {len(raws)} Loaded GIPSA data.")

    all_features, all_labels, all_groups = [], [], []
    for raw, subject_id in raws:
        features, labels = extract_windows_and_labels(raw)  # already per-subject normalized
        all_features.append(features)
        all_labels.append(labels)
        all_groups.append(np.full(len(labels), subject_id))

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    print(f"[pretrain] {X.shape[0]} window total, {X.shape[1]} features/window.")

    np.savez(
        "gipsa_training_features.npz",
        X=X, y=y, groups=groups,
        channel_names=ALPHA_CHANNEL_NAMES,
    )
    print(f"[pretrain] Training features saved to: gipsa_training_features.npz")

    # No StandardScaler anymore in pipeline: X is already per-subject
    # z-normaalized.
    model = LogisticRegression()

    scores = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(n_splits=5))
    print(f"[pretrain] Cross-Subject-Accuracy: {scores.mean():.3f} "
          f"(+/- {scores.std():.3f})")

    model.fit(X, y)
    dump(model, MODEL_OUTPUT_PATH)
    print(f"[pretrain] Model saved: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()