"""
demo_visual.py - lightweight live visualization for the alpha-power demo.
Connects to the AlphaState LSL marker stream (pushed by live_classify.py)
and renders: a big color-coded status indicator, a rolling probability
curve, a per-channel contribution bar chart with the model's static
trained weights as reference, a live feature-space projection against the
labeled GIPSA training cloud, and a raw per-channel feature time series.
A 6th panel slot is reserved, currently empty, for future additions.

Run live_classify.py in one terminal, this script in another.
"""

import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from pylsl import StreamInlet, resolve_byprop
from sklearn.decomposition import PCA

from config import ALPHA_CHANNEL_NAMES

STREAM_NAME = "AlphaState"
HISTORY_SECONDS = 12
RESOLVE_TIMEOUT_S = 5.0
SUSTAINED_CLOSED_THRESHOLD_S = 4.0

MODEL_PATH = "alpha_classifier.joblib"
TRAINING_DATA_PATH = "gipsa_training_features.npz"


def connect_to_state_stream(max_attempts=8):
    for attempt in range(1, max_attempts + 1):
        print(f"[demo] Searching '{STREAM_NAME}' (attempt {attempt}/{max_attempts}) ...")
        results = resolve_byprop("name", STREAM_NAME, timeout=RESOLVE_TIMEOUT_S)
        if results:
            print(f"[demo] connected to '{STREAM_NAME}'")
            return StreamInlet(results[0])
    raise RuntimeError(f"Stream '{STREAM_NAME}' not found - is live_classify.py running?")


def parse_marker(raw_string):
    state, confidence_str, feature_str = raw_string.split("|")
    feature = np.array([float(v) for v in feature_str.split(",")])
    return state, float(confidence_str), feature


def load_explainability_context():
    """Loads everything needed to explain a live decision against the
    model's TRAINING-time behavior: the model's fixed learned weights,
    and a PCA projection fitted on the GIPSA training set to use as a
    static reference cloud.
    """
    model = load(MODEL_PATH)
    weights = model.coef_[0]

    data = np.load(TRAINING_DATA_PATH, allow_pickle=True)
    X_train, y_train = data["X"], data["y"]

    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)

    return weights, pca, X_train_2d, y_train


def main():
    inlet = connect_to_state_stream()
    weights, pca, X_train_2d, y_train = load_explainability_context()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    (ax_indicator, ax_history, ax_contrib), (ax_feature_space, ax_raw_features, ax_reserved) = axes
    ax_reserved.axis("off")  # 6th slot - reserved for future additions
    plt.ion()
    fig.show()

    # -- panel 1: big status rectangle --
    indicator = ax_indicator.add_patch(plt.Rectangle((0, 0), 1, 1, color="gray"))
    ax_indicator.set_xlim(0, 1)
    ax_indicator.set_ylim(0, 1)
    ax_indicator.set_xticks([])
    ax_indicator.set_yticks([])
    status_text = ax_indicator.text(0.5, 0.5, "", ha="center", va="center",
                                     fontsize=16, fontweight="bold", color="white")
    warning_text = ax_indicator.text(0.5, 0.08, "", ha="center", va="center",
                                      fontsize=10, fontweight="bold", color="yellow")

    # -- panel 2: rolling probability curve --
    times, probs = deque(), deque()
    line, = ax_history.plot([], [], color="tab:blue")
    ax_history.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax_history.set_ylim(-0.05, 1.05)
    ax_history.set_xlabel("seconds ago")
    ax_history.set_ylabel("P(eyes closed)")
    ax_history.set_title("Rolling probability")

    # -- panel 3: per-channel contribution, with trained weights as legend --
    bars = ax_contrib.bar(ALPHA_CHANNEL_NAMES, [0, 0, 0, 0], color="gray")
    ax_contrib.axhline(0, color="black", linewidth=0.8)
    ax_contrib.set_title("Live channel contribution\n(weight x current value)")
    ax_contrib.set_ylabel("contribution toward 'closed'")
    weights_legend = "  ".join(f"{name}={w:+.2f}" for name, w in zip(ALPHA_CHANNEL_NAMES, weights))
    ax_contrib.text(0.5, -0.28, f"Trained weights (fixed, from GIPSA):\n{weights_legend}",
                     ha="center", va="top", transform=ax_contrib.transAxes, fontsize=8)

    # -- panel 4: live feature space against the labeled training cloud --
    ax_feature_space.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1],
                              color="tab:red", alpha=0.15, s=15, label="GIPSA: closed (labeled)")
    ax_feature_space.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1],
                              color="tab:green", alpha=0.15, s=15, label="GIPSA: open (labeled)")
    # Neutral color (black) for both live markers - deliberately NOT
    # red/green, since we have no ground truth for live points (unlike
    # the GIPSA cloud). Marker SHAPE instead encodes the model's own
    # guess. All points accumulated this session are shown - no fading.
    live_scatter_closed = ax_feature_space.scatter([], [], color="black", marker="x",
                                                    s=35, linewidth=1.2,
                                                    label="Live - model guess: closed")
    live_scatter_open = ax_feature_space.scatter([], [], color="black", marker="o",
                                                  s=45, linewidth=1.2,
                                                  label="Live - model guess: open")
    ax_feature_space.set_title("Live position in feature space")
    ax_feature_space.set_xlabel("PC1")
    ax_feature_space.set_ylabel("PC2")
    ax_feature_space.legend(loc="upper right", fontsize=7)

    live_points_closed = []
    live_points_open = []

    # -- panel 5: raw per-channel feature over time --
    raw_feature_times = deque()
    raw_feature_values = {name: deque() for name in ALPHA_CHANNEL_NAMES}
    raw_feature_lines = {}
    for name in ALPHA_CHANNEL_NAMES:
        raw_line, = ax_raw_features.plot([], [], label=name)
        raw_feature_lines[name] = raw_line
    ax_raw_features.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_raw_features.set_xlabel("seconds ago")
    ax_raw_features.set_ylabel("baseline z-score")
    ax_raw_features.set_title("Raw per-channel feature\n(what the model actually sees)")
    ax_raw_features.legend(fontsize=8)

    fig.tight_layout(pad=2.5, w_pad=3.0, h_pad=3.0)

    start_time = time.monotonic()
    closed_since = None

    print("[demo] Visualization running. Close the plot window to stop.")
    try:
        while plt.fignum_exists(fig.number):
            sample, _ = inlet.pull_sample(timeout=0.2)
            now = time.monotonic()

            if sample is not None:
                state, confidence, normalized_feature = parse_marker(sample[0])
                is_closed = state == "EYES CLOSED"
                elapsed = now - start_time

                # -- panel 1 --
                color = "tab:red" if is_closed else "tab:green"
                indicator.set_color(color)
                indicator.set_alpha(0.25 + 0.75 * confidence)
                status_text.set_text(f"{state}\n{confidence:.0%}")

                if is_closed:
                    closed_since = closed_since or now
                    warning_text.set_text(
                        "SIMULATED\nTAKEOVER REQUEST"
                        if now - closed_since >= SUSTAINED_CLOSED_THRESHOLD_S else ""
                    )
                else:
                    closed_since = None
                    warning_text.set_text("")

                # -- panel 2 --
                times.append(elapsed)
                probs.append(confidence if is_closed else 1 - confidence)
                while times and elapsed - times[0] > HISTORY_SECONDS:
                    times.popleft()
                    probs.popleft()
                if times:
                    line.set_data([t - elapsed for t in times], probs)
                    ax_history.set_xlim(-HISTORY_SECONDS, 0)

                # -- panel 3 --
                contributions = weights * normalized_feature
                for bar, contribution in zip(bars, contributions):
                    bar.set_height(contribution)
                    bar.set_color("tab:red" if contribution > 0 else "tab:green")
                ax_contrib.relim()
                ax_contrib.autoscale_view()

                # -- panel 4 --
                point_2d = pca.transform(normalized_feature.reshape(1, -1))[0]
                if is_closed:
                    live_points_closed.append(point_2d)
                    live_scatter_closed.set_offsets(np.array(live_points_closed))
                else:
                    live_points_open.append(point_2d)
                    live_scatter_open.set_offsets(np.array(live_points_open))

                # -- panel 5 --
                raw_feature_times.append(elapsed)
                for name, value in zip(ALPHA_CHANNEL_NAMES, normalized_feature):
                    raw_feature_values[name].append(value)
                while raw_feature_times and elapsed - raw_feature_times[0] > HISTORY_SECONDS:
                    raw_feature_times.popleft()
                    for name in ALPHA_CHANNEL_NAMES:
                        raw_feature_values[name].popleft()
                if raw_feature_times:
                    relative_times = [t - elapsed for t in raw_feature_times]
                    for name in ALPHA_CHANNEL_NAMES:
                        raw_feature_lines[name].set_data(relative_times, raw_feature_values[name])
                    ax_raw_features.set_xlim(-HISTORY_SECONDS, 0)
                    ax_raw_features.relim()
                    ax_raw_features.autoscale_view()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[demo] Stopped.")


if __name__ == "__main__":
    main()