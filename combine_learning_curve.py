"""
combine_learning_curve.py
Aggregates individual Slurm array task outputs (learning_curve_fraction_*.csv)
into a master summary table (learning_curve_summary.csv) matching the thesis schema.
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Combine Slurm array learning curve metrics")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing learning_curve_fraction_*.csv files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save combined results and summary CSVs")
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.input_dir, "learning_curve_fraction_*.csv")))
    if not files:
        print(f"[ERROR] No task files found in: {args.input_dir}")
        sys.exit(1)

    print(f"[COMBINE] Located {len(files)} fraction metric files.")
    dfs = [pd.read_csv(f) for f in files]
    all_runs = pd.concat(dfs, ignore_index=True)

    # Save detailed runs
    detailed_file = os.path.join(args.output_dir, "learning_curve_results_all.csv")
    all_runs.to_csv(detailed_file, index=False)
    print(f"[SAVED] Detailed iteration records: {detailed_file}")

    # Compute Summary Statistics by Fraction (Matching Thesis Format)
    summary_rows = []
    for frac, group in all_runs.groupby("fraction"):
        n_reps = len(group)
        acc = group["accuracy"].to_numpy()
        f1 = group["f1_score"].to_numpy()
        prauc = group["pr_auc"].to_numpy()

        summary_rows.append({
            "Prop": frac,
            "Mean_Accuracy": np.mean(acc),
            "SD_Accuracy": np.std(acc, ddof=1) if n_reps > 1 else 0.0,
            "SE_Accuracy": (np.std(acc, ddof=1) / np.sqrt(n_reps)) if n_reps > 1 else 0.0,
            "Min_Accuracy": np.min(acc),
            "Max_Accuracy": np.max(acc),
            "Median_Accuracy": np.median(acc),
            "Mean_F1": np.mean(f1),
            "SD_F1": np.std(f1, ddof=1) if n_reps > 1 else 0.0,
            "Mean_PR_AUC": np.mean(prauc),
            "N_Reps": n_reps,
            "Mean_Roads": int(np.round(group["n_roads"].mean())),
            "Mean_Pixels": int(np.round(group["n_pixels"].mean()))
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Prop").reset_index(drop=True)
    summary_file = os.path.join(args.output_dir, "learning_curve_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"[SAVED] Publication Summary Table: {summary_file}")
    print("\nSummary Results:")
    print(summary_df[["Prop", "Mean_Accuracy", "SD_Accuracy", "Mean_F1", "Mean_Roads", "Mean_Pixels"]].to_string(index=False))


if __name__ == "__main__":
    main()
