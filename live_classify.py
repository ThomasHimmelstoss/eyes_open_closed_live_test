"""
live_classify.py

combines LSL connection, feature extraction and individual subject-based baseline normalization
together with the pretrained GIPSA model for a live prediction on alpha power

requirements for running live_classify.py: calibration.py was executed for this specific person/ session 
(baseline_calibration.joblib exists) and pretrain_gipsa.py was executed once (alpha_classifier.joblib exists)
"""

import csv
import os
from collections import deque
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet

from joblib import load

from config import (ALPHA_CHANNEL_INDICES, ALPHA_CHANNEL_NAMES, RECORDINGS_DIR,
                     STEP_SECONDS_LIVE, SUBJECT_ID, WINDOW_SECONDS)
from lsl_stream import connect_inlet, window_generator
from relative_alpha_power_features import relative_alpha_power

MODEL_PATH = "alpha_classifier.joblib"  # global, not per-subject - trained once on GIPSA


# majority smoothing over n amount of predictions
# e.g. STEÜ_SECONDS_LIVE=1s & SMOOTHING_WINDOW=3 = 3s "delay" until change of alpha power is displayed
# set to "1" to deactivate smoothing
SMOOTHING_WINDOW = 3

def find_latest_session_dir(subject_id):
    """Finds the most recently created calibration session folder for this
    subject (folder names are timestamps, so lexicographic == chronological
    order - see calibration.py's datetime.now().strftime("%Y-%m-%d_%H%M%S")).
    """
    subject_dir = os.path.join(RECORDINGS_DIR, subject_id)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(
            f"No recordings found for subject '{subject_id}' under "
            f"'{subject_dir}'. Run calibration.py first."
        )

    session_names = sorted(os.listdir(subject_dir))
    if not session_names:
        raise FileNotFoundError(f"'{subject_dir}' exists but contains no sessions.")

    return os.path.join(subject_dir, session_names[-1])


def load_artifacts(session_dir):
    model = load(MODEL_PATH)
    baseline = load(os.path.join(session_dir, "baseline_calibration.joblib"))
    return model, baseline



def normalize_against_baseline(feature_vector, baseline):
    """Subject-individual Z-Score against subject-specific baseline distribution.

    this normalization approach is identical to what was performed on the GIPSA training data.
    """
    return (feature_vector - baseline["mean"]) / baseline["std"]


def main():
    session_dir = find_latest_session_dir(SUBJECT_ID)
    print(f"[live] using calibration session: {session_dir}")

    model, baseline = load_artifacts(session_dir)
    weights = model.coef_[0]  # shape (4,) - ein Gewicht pro Kanal, in der Reihenfolge von ALPHA_CHANNEL_NAMES
    bias = model.intercept_[0]
    inlet = connect_inlet()

    state_info = StreamInfo("AlphaState", "Markers", 1, 0, "string", "alpha_state_outlet")
    state_outlet = StreamOutlet(state_info)

    log_path = os.path.join(session_dir, "live_session_log.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    # raw per-window features (baseline-normalized) are logged alongside the
    # aggregated prob/state columns, so a later post-hoc run can explore the
    # feature space itself, not just the model's already-smoothed verdict.
    log_writer.writerow(["timestamp", *ALPHA_CHANNEL_NAMES, "prob_closed_raw",
                          "prob_closed_smoothed", "state"])

    recent_probabilities = deque(maxlen=SMOOTHING_WINDOW)

    print("[live] Classification running. Ctrl+C to stop.")
    try:
        for samples, _ in window_generator(inlet, window_seconds=WINDOW_SECONDS,
                                            step_seconds=STEP_SECONDS_LIVE):
            eeg_window = samples[:, ALPHA_CHANNEL_INDICES]
            raw_feature = relative_alpha_power(eeg_window)
            normalized_feature = normalize_against_baseline(raw_feature, baseline)

            contributions = weights * normalized_feature  # elementweise: Beitrag jedes Kanals zur Summe

            probability = model.predict_proba(normalized_feature.reshape(1, -1))[0]
            prob_closed = probability[1]  # P(closed) - model.classes_ confirmed [0, 1]

            recent_probabilities.append(prob_closed)
            smoothed_prob_closed = sum(recent_probabilities) / len(recent_probabilities)

            state = "EYES CLOSED" if smoothed_prob_closed >= 0.5 else "EYES OPEN"
            confidence = smoothed_prob_closed if smoothed_prob_closed >= 0.5 else 1 - smoothed_prob_closed

            # push state, confidence and features per channel to outlet for "adaptive system"
            state_outlet.push_sample([f"{state}|{confidence:.3f}|{','.join(f'{c:.3f}' for c in contributions)}"])

            print(f"[live] {state}  (confidence: {confidence:.2f})")
            log_writer.writerow([
                datetime.now().isoformat(),
                *normalized_feature,          # P3, P4, O1, O2 - baseline-normalized
                f"{prob_closed:.4f}",         # this single window's model probability
                f"{smoothed_prob_closed:.4f}",
                state,
            ])
            log_file.flush()  # write immediately, so a crash/Ctrl+C never loses buffered rows

    except KeyboardInterrupt:
        print("\n[live] Stopped.")
    finally:
        log_file.close()


if __name__ == "__main__":
    main()