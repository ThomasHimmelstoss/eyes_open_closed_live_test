"""
analyze_features.py - post-hoc analysis of eyes-closed vs. eyes-open
alpha-power features, from three possible sources:
  - a calibration session (recordings/<subject>/<session>/calibration_raw.npz)
  - the GIPSA training set (gipsa_training_features.npz)
  - a live-test log (recordings/<subject>/<session>/live_session_log.csv),
    analyzed UNSUPERVISED since there's no reliable ground-truth condition
    per window - shows the feature-space structure and how well it lines
    up with the model's own (smoothed) decision.

Usage:
  python analyze_features.py calibration <session_dir>
  python analyze_features.py training gipsa_training_features.npz
  python analyze_features.py live <session_dir>/live_session_log.csv
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests 

def load_calibration(session_dir):
    data = np.load(f"{session_dir}/calibration_raw.npz", allow_pickle=True)
    return data["closed_features"], data["open_features"], list(data["channel_names"])


def load_training(npz_path):
    """GIPSA training data has one combined X/y array (many subjects
    pooled) instead of two pre-split arrays - split it here so the rest
    of the analysis code doesn't need to know the difference.
    """
    data = np.load(npz_path, allow_pickle=True)
    X, y, channel_names = data["X"], data["y"], list(data["channel_names"])
    closed_features = X[y == 1]
    open_features = X[y == 0]
    return closed_features, open_features, channel_names


def cohens_d(a, b):
    """Standardized effect size - how many pooled standard deviations
    apart the two condition means are. Independent of the raw feature
    scale, so it's comparable across channels and sessions.
    """
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def summarize_channel(closed, open_, name):
    # Mann-Whitney U: doesn't assume normally distributed features - more
    # robust than a t-test for small/skewed samples (e.g. a single
    # ~25s calibration block gives only ~20 windows per side).
    u_stat, p_value = stats.mannwhitneyu(closed, open_, alternative="two-sided")
    d = cohens_d(closed, open_)

    print(f"{name:>4s}:  closed_mean={closed.mean():+.3f}  open_mean={open_.mean():+.3f}  "
          f"diff={closed.mean() - open_.mean():+.3f}  cohens_d={d:+.2f}  p={p_value:.4f}")

    return {"channel": name, "diff": closed.mean() - open_.mean(), "cohens_d": d, "p_value": p_value}

def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def plot_condition_comparison(closed_features, open_features, channel_names, results, output_path, title):
    n_channels = len(channel_names)
    fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4), sharey=True)
    if n_channels == 1:
        axes = [axes]

    for i, (ax, name, result) in enumerate(zip(axes, channel_names, results)):
        closed_col = closed_features[:, i]
        open_col = open_features[:, i]

        ax.boxplot([closed_col, open_col], tick_labels=["closed", "open"])
        ax.scatter(np.full(len(closed_col), 1), closed_col, alpha=0.3, color="tab:blue")
        ax.scatter(np.full(len(open_col), 2), open_col, alpha=0.3, color="tab:orange")

        # Significance annotation: bracket + stars above both boxes
        y_max = max(closed_col.max(), open_col.max())
        y_range = max(closed_col.max(), open_col.max()) - min(closed_col.min(), open_col.min())
        bracket_y = y_max + 0.05 * y_range
        ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + 0.02 * y_range,
                                bracket_y + 0.02 * y_range, bracket_y], color="black", linewidth=1)
        ax.text(1.5, bracket_y + 0.03 * y_range, significance_stars(result["p_value_corrected"]),
                ha="center", va="bottom", fontsize=12)

        ax.set_title(name)
        if i == 0:
            ax.set_ylabel("log relative alpha power")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


def analyze_two_conditions(closed_features, open_features, channel_names, output_path, title):
    print(f"Windows: {len(closed_features)} closed, {len(open_features)} open\n")

    results = [
        summarize_channel(closed_features[:, i], open_features[:, i], name)
        for i, name in enumerate(channel_names)
    ]

    # Multiple-comparisons correction across the 4 channel tests -
    # Holm-Bonferroni: same family-wise error control as plain Bonferroni,
    # slightly more statistical power (rarely worse, never more liberal).
    raw_p_values = [r["p_value"] for r in results]
    reject, corrected_p_values, _, _ = multipletests(raw_p_values, alpha=0.05, method="holm")
    for result, corrected_p, is_significant in zip(results, corrected_p_values, reject):
        result["p_value_corrected"] = corrected_p
        result["significant_corrected"] = is_significant

    print()
    for r in results:
        marker = "*" if r["significant_corrected"] else " "
        print(f"{marker} {r['channel']:>4s}:  p_raw={r['p_value']:.4f}  "
              f"p_holm={r['p_value_corrected']:.4f}")

    best = max(results, key=lambda r: abs(r["cohens_d"]))
    print(f"\nStrongest effect: {best['channel']} (cohens_d={best['cohens_d']:+.2f}, "
          f"p_holm={best['p_value_corrected']:.4f})")

    plot_condition_comparison(closed_features, open_features, channel_names, results, output_path, title)


def load_live_log(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    channel_names = ["P3", "P4", "O1", "O2"]
    X = df[channel_names].to_numpy()
    return df, X, channel_names


def analyze_live_log(csv_path):
    df, X, channel_names = load_live_log(csv_path)
    print(f"Windows: {len(df)}")
    print(f"Time span: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Unsupervised: does the feature space naturally separate into 2
    # clusters, and if so, do they line up with what the model already
    # decided? This is exploratory - it does NOT assume the clusters
    # correspond to the true eyes-open/closed labels, since we have no
    # ground truth here.
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(X)

    model_state_binary = (df["state"] == "EYES CLOSED").astype(int).to_numpy()
    agreement = max(
        (cluster_labels == model_state_binary).mean(),
        (cluster_labels == 1 - model_state_binary).mean(),  # cluster IDs are arbitrary, check both mappings
    )

    # K-means cluster looks at the data "unsupervised" - so without knowledge of any true labels - and 
    # groups the data into 2 separatable clouds -> then this clustering is compared to what the actual 
    # model predicted during the live test -> higher agreement confirms that both approaches 
    # manage to indentify a specific 2-groups-structure within the 4 dimensional feature space
    # (not really a validation, both classification approaches might fall for similar artifacts and display unrelated agreement
    # - just helps to confirm that there is some grouped data within)
    print(f"Unsupervised 2-cluster agreement with model's smoothed state: {agreement:.1%}")

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # PCA combines/ compresses the 4D-feature space into a 2D-feature space. This comes with a certain loss in information traded for 
    # visualizing the feature space in 2D. pca.explained_variance_ratio_sum shows the amount of variability that is still
    # present within the 2D plane that was originally present within the 4D space
    # If all 4 features/ channels would be uncorrelated -> then 2 out of 4 channels would only be able to explain 50% of the variance
    # If we see higher than 50% variance -> indicates that channels are correlated to each other and feature values 
    # move im similar directions
    # Ulitmately the explained variance ratio tells us later on in the PCA plot how much we can trust what we are seeing there.
    print(f"PCA explained variance (2 components): {pca.explained_variance_ratio_.sum():.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Time series of the model's own smoothed probability
    axes[0].plot(df["timestamp"], df["prob_closed_smoothed"], color="tab:blue")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("smoothed P(eyes closed)")
    axes[0].set_title("Model output over time")
    axes[0].tick_params(axis="x", rotation=30)

    # (b) Per-channel feature distributions - is there any bimodality at all?
    axes[1].boxplot([X[:, i] for i in range(len(channel_names))], tick_labels=channel_names)
    axes[1].set_ylabel("baseline-normalized feature (z-score)")
    axes[1].set_title("Per-channel distribution (unlabeled)")

    # (c) 2D PCA scatter, colored by the unsupervised cluster assignment
    scatter = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="coolwarm", alpha=0.6)
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title("Feature space (colored by k-means cluster)")

    fig.tight_layout()
    output_path = csv_path.replace(".csv", "_exploration.png")
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    mode, path = sys.argv[1], sys.argv[2]

    if mode == "calibration":
        closed, open_, names = load_calibration(path)
        analyze_two_conditions(closed, open_, names, f"{path}/comparison_plot.png",
                                "Calibration - eyes closed vs. eyes open")
    elif mode == "training":
        closed, open_, names = load_training(path)
        analyze_two_conditions(closed, open_, names, "gipsa_training_comparison.png",
                                "GIPSA training set - eyes closed vs. eyes open")
    elif mode == "live":
        analyze_live_log(path)
    else:
        print(f"Unknown mode '{mode}'. Use 'calibration', 'training', or 'live'.")
        sys.exit(1)


if __name__ == "__main__":
    main()