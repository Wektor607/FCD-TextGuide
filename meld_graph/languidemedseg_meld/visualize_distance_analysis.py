"""
Visualize distance head cluster analysis.
Compares aggregation methods (min, mean, median) and shows distributions for TP vs FP.

Usage:
    python visualize_distance_analysis.py \
        --csv_main save_model/exp3_..._MELD_..._cluster_dist_analysis.csv \
        --csv_indep save_model/exp3_..._BONN_..._cluster_dist_analysis.csv \
        --output_dir plots/
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def parse_is_control(val):
    """Parse is_control column which may contain 'tensor(True)' strings."""
    if isinstance(val, bool):
        return val
    s = str(val).lower().strip()
    return "true" in s


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df["is_control"] = df["is_control"].apply(parse_is_control)
    return df


def plot_distributions(df, cohort_name, output_dir):
    """Box plots of TP vs FP vs FP(ctrl) for each aggregation method."""
    tp = df[df["label"] == "TP"]
    fp_patient = df[(df["label"] == "FP") & (~df["is_control"])]
    fp_ctrl = df[(df["label"] == "FP") & (df["is_control"])]

    agg_cols = ["pred_dist_min", "pred_dist_mean", "pred_dist_median"]
    agg_labels = ["min", "mean", "median"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Distance Head Predictions by Cluster Type — {cohort_name}", fontsize=14)

    for ax, col, label in zip(axes, agg_cols, agg_labels):
        data = []
        labels_list = []
        colors = []

        if len(tp) > 0:
            data.append(tp[col].values)
            labels_list.append(f"TP\n(n={len(tp)})")
            colors.append("#2196F3")
        if len(fp_patient) > 0:
            data.append(fp_patient[col].values)
            labels_list.append(f"FP (patient)\n(n={len(fp_patient)})")
            colors.append("#FF9800")
        if len(fp_ctrl) > 0:
            data.append(fp_ctrl[col].values)
            labels_list.append(f"FP (control)\n(n={len(fp_ctrl)})")
            colors.append("#F44336")

        bp = ax.boxplot(data, labels=labels_list, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Predicted distance (normalized)")
        ax.set_title(f"Aggregation: {label}")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"dist_distributions_{cohort_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_histograms_overlay(df, cohort_name, output_dir):
    """Overlaid histograms of TP vs FP for each aggregation."""
    tp = df[df["label"] == "TP"]
    fp = df[df["label"] == "FP"]

    agg_cols = ["pred_dist_min", "pred_dist_mean", "pred_dist_median"]
    agg_labels = ["min", "mean", "median"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Distribution of Predicted Distances — {cohort_name}", fontsize=14)

    for ax, col, label in zip(axes, agg_cols, agg_labels):
        bins = np.linspace(-1.5, 1.5, 50)
        ax.hist(tp[col].values, bins=bins, alpha=0.6, color="#2196F3",
                label=f"TP (n={len(tp)})", density=True)
        ax.hist(fp[col].values, bins=bins, alpha=0.6, color="#F44336",
                label=f"FP (n={len(fp)})", density=True)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Predicted distance (normalized)")
        ax.set_ylabel("Density")
        ax.set_title(f"Aggregation: {label}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"dist_histograms_{cohort_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def compute_sweep(df, agg_col, thresholds):
    """Compute sensitivity/specificity for a given aggregation at each threshold."""
    tp_rows = df[df["label"] == "TP"]
    fp_ctrl = df[(df["label"] == "FP") & (df["is_control"])]

    total_patients = df[df["label"] == "TP"]["subject_id"].nunique()
    ctrl_subjects = df[df["is_control"]]["subject_id"].unique()
    total_ctrl = len(ctrl_subjects)

    sensitivities = []
    specificities = []

    for thresh in thresholds:
        # Sensitivity: patients with at least one surviving TP cluster
        tp_surviving = tp_rows[tp_rows[agg_col] < thresh]
        patients_with_tp = tp_surviving["subject_id"].nunique() if len(tp_surviving) > 0 else 0
        sens = patients_with_tp / total_patients if total_patients > 0 else 0.0

        # Specificity: controls with zero surviving FP clusters
        fp_ctrl_surviving = fp_ctrl[fp_ctrl[agg_col] < thresh]
        ctrl_with_fp = fp_ctrl_surviving["subject_id"].nunique() if len(fp_ctrl_surviving) > 0 else 0
        spec = (total_ctrl - ctrl_with_fp) / total_ctrl if total_ctrl > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)

    return np.array(sensitivities), np.array(specificities)


def plot_sweep_comparison(df, cohort_name, output_dir):
    """Compare sweep curves for different aggregation methods."""
    agg_cols = ["pred_dist_min", "pred_dist_mean", "pred_dist_median"]
    agg_labels = ["min", "mean", "median"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    thresholds = np.arange(-0.2, 0.8, 0.02)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Filter Sweep by Aggregation Method — {cohort_name}", fontsize=14)

    for col, label, color in zip(agg_cols, agg_labels, colors):
        sens, spec = compute_sweep(df, col, thresholds)
        axes[0].plot(thresholds, sens, label=label, color=color, linewidth=2)
        axes[1].plot(thresholds, spec, label=label, color=color, linewidth=2)

    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("Sensitivity vs Threshold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Specificity")
    axes[1].set_title("Specificity vs Threshold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, f"dist_sweep_{cohort_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_sens_vs_spec(df, cohort_name, output_dir):
    """Sensitivity vs Specificity trade-off curve for each aggregation."""
    agg_cols = ["pred_dist_min", "pred_dist_mean", "pred_dist_median"]
    agg_labels = ["min", "mean", "median"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    thresholds = np.arange(-0.5, 1.0, 0.01)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(f"Sensitivity vs Specificity — {cohort_name}", fontsize=14)

    for col, label, color in zip(agg_cols, agg_labels, colors):
        sens, spec = compute_sweep(df, col, thresholds)
        ax.plot(1 - spec, sens, label=label, color=color, linewidth=2)

        # Mark Youden's J optimal point
        j_index = sens + spec - 1
        best_idx = np.argmax(j_index)
        ax.scatter(1 - spec[best_idx], sens[best_idx], color=color, s=100, zorder=5,
                   edgecolors="black", linewidths=1.5)
        ax.annotate(f"J={j_index[best_idx]:.2f}\nthr={thresholds[best_idx]:.2f}",
                    xy=(1 - spec[best_idx], sens[best_idx]),
                    xytext=(15, -15), textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=color))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(output_dir, f"dist_sens_vs_spec_{cohort_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_optimal_thresholds(df, cohort_name):
    """Print optimal thresholds for each aggregation using Youden's J."""
    agg_cols = ["pred_dist_min", "pred_dist_mean", "pred_dist_median"]
    agg_labels = ["min", "mean", "median"]

    thresholds = np.arange(-0.5, 1.0, 0.01)

    print(f"\n=== OPTIMAL THRESHOLDS (Youden's J) — {cohort_name} ===")
    print(f"{'Aggregation':>12} | {'Threshold':>9} | {'Sensitivity':>11} | {'Specificity':>11} | {'Youden J':>8}")
    print("-" * 65)

    for col, label in zip(agg_cols, agg_labels):
        sens, spec = compute_sweep(df, col, thresholds)
        j_index = sens + spec - 1
        best_idx = np.argmax(j_index)
        print(f"{label:>12} | {thresholds[best_idx]:>9.2f} | {sens[best_idx]:>10.1%} | {spec[best_idx]:>10.1%} | {j_index[best_idx]:>8.3f}")


def extract_name_from_csv(csv_path):
    """Extract unique experiment name from CSV filename.
    e.g. '...exp3_mixed_no_gnn_radbert_BONN_MELD_BONN_hemisphere_lobe_text_cluster_dist_analysis.csv'
      -> 'BONN_MELD_BONN_hemisphere_lobe_text'
    """
    basename = os.path.basename(csv_path)
    # strip suffix
    name = basename.replace("_cluster_dist_analysis.csv", "")
    # strip common prefix
    for prefix in ["exp3_mixed_no_gnn_radbert_", "exp3_mixed_no_gnn_", "exp3_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_main", type=str, default=None, help="Main cohort CSV")
    parser.add_argument("--csv_indep", type=str, default=None, help="Independent cohort CSV")
    parser.add_argument("--output_dir", type=str, default="plots/distance_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = []
    if args.csv_main:
        datasets.append((args.csv_main, extract_name_from_csv(args.csv_main)))
    if args.csv_indep:
        datasets.append((args.csv_indep, extract_name_from_csv(args.csv_indep)))

    if not datasets:
        print("No CSV files provided. Use --csv_main and/or --csv_indep.")
        return

    for csv_path, name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {name} ({csv_path})")
        print(f"{'='*60}")

        df = load_and_clean(csv_path)

        tp_count = len(df[df["label"] == "TP"])
        fp_count = len(df[df["label"] == "FP"])
        fp_ctrl = len(df[(df["label"] == "FP") & (df["is_control"])])
        print(f"Clusters: {len(df)} total (TP={tp_count}, FP={fp_count}, FP on controls={fp_ctrl})")

        plot_distributions(df, name, args.output_dir)
        plot_histograms_overlay(df, name, args.output_dir)
        plot_sweep_comparison(df, name, args.output_dir)
        plot_sens_vs_spec(df, name, args.output_dir)
        print_optimal_thresholds(df, name)


if __name__ == "__main__":
    main()
