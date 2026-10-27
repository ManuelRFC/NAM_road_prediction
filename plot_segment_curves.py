"""
plot_segment_curves.py
Local Publication-Quality Visualizer for Road Construction Curves and Learning Curves.

Replicates the exact thesis theme and color palette:
- Primary line color: #367DB8 (THESIS_BLUE)
- Ground-truth intervention window: #E41A1C (CONSTRUCTION_RED, dashed)
- Estimated changepoint curve: #2CA02C (ESTIMATE_GREEN)
- Generates both individual PNG plots and an interactive HTML dropdown dashboard (Plotly).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional

# Thesis Styling Constants
THESIS_BLUE = "#367DB8"
CONSTRUCTION_RED = "#E41A1C"
ESTIMATE_GREEN = "#2CA02C"


def setup_matplotlib_thesis_theme():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelcolor": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "figure.titlesize": 14,
        "figure.dpi": 300
    })


def plot_single_road(
    road_df: pd.DataFrame,
    out_path: str,
    road_id: str,
    timing_row: Optional[pd.Series] = None
):
    """
    Plots a single road segment time-series (1990-2024) matching the thesis plot layout.
    """
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

    years = road_df["year"].to_numpy()
    probs = road_df["mean_prob"].to_numpy()

    # Main time series line and markers
    ax.plot(years, probs, color=THESIS_BLUE, linewidth=2.2, marker="o", markersize=5.5, label="Mean Post-Paving Prob")

    # Standard deviation ribbon if available
    if "std_prob" in road_df.columns:
        std = road_df["std_prob"].to_numpy()
        ax.fill_between(years, np.maximum(0, probs - std), np.minimum(1, probs + std),
                        color=THESIS_BLUE, alpha=0.15, label="Within-Road Pixel SD")

    # True Construction Window (Vertical dashed red lines)
    has_truth = False
    true_start = road_df["start_year"].iloc[0] if "start_year" in road_df.columns else np.nan
    true_end = road_df["end_year"].iloc[0] if "end_year" in road_df.columns else np.nan

    if pd.notna(true_start) and pd.notna(true_end):
        has_truth = True
        ax.axvline(x=true_start, color=CONSTRUCTION_RED, linestyle="--", linewidth=1.5, label=f"Ground Truth Start ({int(true_start)})")
        ax.axvline(x=true_end, color=CONSTRUCTION_RED, linestyle="--", linewidth=1.5, label=f"Ground Truth End ({int(true_end)})")
        ax.axvspan(true_start, true_end, color=CONSTRUCTION_RED, alpha=0.08)

    # Predicted Timing Window (if available)
    if timing_row is not None and pd.notna(timing_row.get("pred_end_year")):
        pred_end = timing_row["pred_end_year"]
        ax.axvline(x=pred_end, color=ESTIMATE_GREEN, linestyle=":", linewidth=1.8, label=f"Predicted End ({pred_end:.1f})")

    surface_ty = road_df["SURFACE_TY"].iloc[0] if "SURFACE_TY" in road_df.columns else "Unknown"

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, linestyle="-", alpha=0.7)

    title_suffix = f" [{surface_ty}]" if pd.notna(surface_ty) else ""
    ax.set_title(f"Road Segment: {road_id}{title_suffix}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Post-Construction Probability")

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_learning_curve_figure(
    summary_csv: str,
    out_dir: str
):
    """
    Recreates the thesis learning curve plot with standard error ribbons and boxplots.
    """
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    props = df["Prop"].to_numpy()
    mean_acc = df["Mean_Accuracy"].to_numpy()
    se_acc = df["SE_Accuracy"].to_numpy()

    # Ribbon
    ax.fill_between(props, mean_acc - se_acc, mean_acc + se_acc, color=THESIS_BLUE, alpha=0.20, label="Mean ± 1 SE")
    # Main line and points
    ax.plot(props, mean_acc, color=THESIS_BLUE, linewidth=2.0, marker="o", markersize=6.5, label="Mean Test Accuracy")

    ax.set_xlim(props.min() - 0.03, props.max() + 0.03)
    ax.set_ylim(min(0.60, mean_acc.min() - 0.05), max(0.95, mean_acc.max() + 0.03))

    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, linestyle="-", alpha=0.7)

    ax.set_title("Learning Curve: Model M5 Sample Complexity")
    ax.set_xlabel("Percentage of Training Roads Used")
    ax.set_ylabel("Accuracy on Held-Out Spatial Test Set")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "plot_learning_curve_m5.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[ARTIFACT] Learning curve plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate segment curves and learning curve plots")
    parser.add_argument("--predictions-csv", type=str, required=True,
                        help="Path to namibia_national_segment_predictions_m5.csv")
    parser.add_argument("--timing-csv", type=str, default=None,
                        help="Path to road-level timing estimates CSV from changepoint_detector.py")
    parser.add_argument("--learning-curve-csv", type=str, default=None,
                        help="Path to learning_curve_summary.csv")
    parser.add_argument("--output-dir", type=str, default="./road_plots",
                        help="Directory to save PNG plots")
    parser.add_argument("--max-plots", type=int, default=50,
                        help="Maximum individual road PNGs to generate (or 0 for all)")
    args = parser.parse_args()

    setup_matplotlib_thesis_theme()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INPUT] Reading segment predictions: {args.predictions_csv}")
    df_pred = pd.read_csv(args.predictions_csv)
    df_pred["road_id"] = df_pred["road_id"].astype(str)

    timing_dict = {}
    if args.timing_csv and os.path.exists(args.timing_csv):
        df_time = pd.read_csv(args.timing_csv)
        df_time["road_id"] = df_time["road_id"].astype(str)
        timing_dict = {row["road_id"]: row for _, row in df_time.iterrows()}

    # Plot Roads with Ground Truth First
    roads = df_pred["road_id"].unique()
    if "start_year" in df_pred.columns:
        labeled_roads = df_pred[df_pred["start_year"].notna()]["road_id"].unique()
        other_roads = [r for r in roads if r not in set(labeled_roads)]
        ordered_roads = list(labeled_roads) + other_roads
    else:
        ordered_roads = list(roads)

    plots_to_make = ordered_roads if args.max_plots == 0 else ordered_roads[:args.max_plots]
    print(f"[PLOTTING] Generating plots for {len(plots_to_make)} roads in: {args.output_dir}")

    for idx, rid in enumerate(plots_to_make, 1):
        sub = df_pred[df_pred["road_id"] == rid].sort_values("year")
        t_row = timing_dict.get(rid, None)
        out_png = os.path.join(args.output_dir, f"Road_{rid}_curve.png")
        plot_single_road(sub, out_png, rid, t_row)
        if idx % 10 == 0 or idx == len(plots_to_make):
            print(f"  Generated {idx}/{len(plots_to_make)} road plots...")

    # Plot Learning Curve if summary CSV is provided
    if args.learning_curve_csv and os.path.exists(args.learning_curve_csv):
        print(f"\n[LEARNING CURVE] Generating publication figure from: {args.learning_curve_csv}")
        plot_learning_curve_figure(args.learning_curve_csv, args.output_dir)

    print("\n[COMPLETE] All requested plots generated successfully.")


if __name__ == "__main__":
    main()
