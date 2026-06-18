"""
Visualize pixel-wise gating sweep results.
Reads CSV files produced by pixelwise_gating_sweep() in test_Kfold.py.

Usage:
    python visualize_pixelwise_gating.py \
        --csv save_model/exp3_..._pixelwise_gating_sweep.csv \
        --csv save_model/exp3_..._pixelwise_gating_sweep.csv \
        --cluster_csv save_model/exp3_..._cluster_dist_analysis.csv \
        --output_dir plots/pixelwise_gating
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_name(csv_path):
    """Extract experiment name from CSV filename."""
    basename = os.path.basename(csv_path)
    name = basename.replace("_pixelwise_gating_sweep.csv", "")
    name = name.replace("_cluster_dist_analysis.csv", "")
    for prefix in ["exp3_mixed_no_gnn_radbert_", "exp3_mixed_no_gnn_", "exp3_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def plot_heatmap(df, name, output_dir):
    """Heatmap of Youden's J over (threshold, temperature) grid."""
    temps = sorted(df["temperature"].unique())
    threshs = sorted(df["threshold"].unique())

    j_matrix = np.full((len(temps), len(threshs)), np.nan)
    for i, t in enumerate(temps):
        for j, th in enumerate(threshs):
            row = df[(df["temperature"] == t) & (df["threshold"] == th)]
            if len(row) > 0:
                j_matrix[i, j] = row["youden_j"].values[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(j_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(threshs)))
    ax.set_xticklabels([f"{th:.2f}" for th in threshs], rotation=45, ha="right")
    ax.set_yticks(range(len(temps)))
    ax.set_yticklabels([f"{t:.2f}" for t in temps])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Youden's J — Pixel-wise Gating — {name}")

    # Annotate cells
    for i in range(len(temps)):
        for j in range(len(threshs)):
            val = j_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    # Mark best cell
    best_idx = np.unravel_index(np.nanargmax(j_matrix), j_matrix.shape)
    ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                                fill=False, edgecolor="red", linewidth=3))

    plt.colorbar(im, ax=ax, label="Youden's J")
    plt.tight_layout()
    path = os.path.join(output_dir, f"pixelwise_heatmap_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_curves_per_temp(df, name, output_dir):
    """Sensitivity and specificity curves for each temperature."""
    temps = sorted(df["temperature"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(temps)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Pixel-wise Gating Sweep — {name}", fontsize=14)

    for temp, color in zip(temps, colors):
        sub = df[df["temperature"] == temp].sort_values("threshold")
        axes[0].plot(sub["threshold"], sub["sensitivity"], color=color,
                     label=f"T={temp:.2f}", linewidth=2)
        axes[1].plot(sub["threshold"], sub["specificity"], color=color,
                     label=f"T={temp:.2f}", linewidth=2)
        axes[2].plot(sub["threshold"], sub["youden_j"], color=color,
                     label=f"T={temp:.2f}", linewidth=2)

    for ax, title, ylabel in zip(axes,
                                  ["Sensitivity", "Specificity", "Youden's J"],
                                  ["Sensitivity", "Specificity", "Youden's J"]):
        ax.set_xlabel("Threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, f"pixelwise_curves_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_roc_per_temp(df, name, output_dir):
    """ROC-like curves (1-Specificity vs Sensitivity) for each temperature."""
    temps = sorted(df["temperature"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(temps)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(f"Sensitivity vs Specificity — Pixel-wise Gating — {name}", fontsize=13)

    for temp, color in zip(temps, colors):
        sub = df[df["temperature"] == temp].sort_values("threshold")
        fpr = 1 - sub["specificity"].values
        tpr = sub["sensitivity"].values
        ax.plot(fpr, tpr, color=color, label=f"T={temp:.2f}", linewidth=2)

        # Mark best Youden's J point
        j_vals = sub["youden_j"].values
        best_i = np.argmax(j_vals)
        ax.scatter(fpr[best_i], tpr[best_i], color=color, s=80, zorder=5,
                   edgecolors="black", linewidths=1.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(output_dir, f"pixelwise_roc_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_best_per_temp(df, name):
    """Print best threshold per temperature."""
    temps = sorted(df["temperature"].unique())

    print(f"\n=== BEST PIXEL-WISE GATING — {name} ===")
    print(f"{'temp':>6} | {'threshold':>9} | {'Sensitivity':>11} | {'Specificity':>11} | {'Youden J':>8}")
    print("-" * 55)
    for temp in temps:
        sub = df[df["temperature"] == temp]
        best = sub.loc[sub["youden_j"].idxmax()]
        print(f"{best['temperature']:>6.2f} | {best['threshold']:>9.2f} | "
              f"{best['sensitivity']:>10.1%} | {best['specificity']:>10.1%} | "
              f"{best['youden_j']:>8.3f}")

    best_overall = df.loc[df["youden_j"].idxmax()]
    print(f"\nBEST OVERALL: thresh={best_overall['threshold']:.2f}, "
          f"temp={best_overall['temperature']:.2f}, "
          f"Sens={best_overall['sensitivity']:.1%}, "
          f"Spec={best_overall['specificity']:.1%}, "
          f"J={best_overall['youden_j']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, nargs="+", required=True,
                        help="Pixel-wise gating sweep CSV(s)")
    parser.add_argument("--output_dir", type=str, default="plots/pixelwise_gating")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for csv_path in args.csv:
        name = extract_name(csv_path)
        print(f"\n{'='*60}")
        print(f"Processing: {name} ({csv_path})")
        print(f"{'='*60}")

        df = pd.read_csv(csv_path)
        print(f"Grid: {df['temperature'].nunique()} temperatures x "
              f"{df['threshold'].nunique()} thresholds = {len(df)} combinations")

        plot_heatmap(df, name, args.output_dir)
        plot_curves_per_temp(df, name, args.output_dir)
        plot_roc_per_temp(df, name, args.output_dir)
        print_best_per_temp(df, name)


if __name__ == "__main__":
    main()
