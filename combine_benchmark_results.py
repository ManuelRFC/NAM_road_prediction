"""
combine_benchmark_results.py
Aggregates Model M1-M7 benchmark metrics across all 5 cross-validation seeds.
Generates publication-ready summary tables (CSV, Markdown, and LaTeX).
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate M1-M7 multi-seed benchmark metrics")
    parser.add_argument("--metrics-dir", type=str, required=True,
                        help="Directory containing metrics_*.json files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save combined benchmark tables")
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.metrics_dir, "metrics_*_national_seed_*.json")))
    if not files:
        # Fall back to any metrics JSON in dir
        files = sorted(glob.glob(os.path.join(args.metrics_dir, "*.json")))

    if not files:
        print(f"[ERROR] No metric files found in: {args.metrics_dir}")
        sys.exit(1)

    print(f"[COMBINE] Found {len(files)} benchmark metric JSON files.")
    records = []
    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
            records.append(data)

    df_raw = pd.DataFrame(records)
    os.makedirs(args.output_dir, exist_ok=True)
    raw_csv = os.path.join(args.output_dir, "benchmark_m1_m7_all_seeds_raw.csv")
    df_raw.to_csv(raw_csv, index=False)
    print(f"[SAVED] Raw seed runs saved to: {raw_csv}")

    # Model ordering
    model_order = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    df_raw["model_upper"] = df_raw["model"].str.upper()

    summary_rows = []
    for m in model_order:
        m_df = df_raw[df_raw["model_upper"] == m]
        if len(m_df) == 0:
            continue

        n_seeds = len(m_df)
        n_feats = m_df["n_features"].iloc[0] if "n_features" in m_df.columns else "N/A"

        acc_vals = m_df["accuracy_mean"].to_numpy()
        f1_vals = m_df["f1_mean"].to_numpy()
        prec_vals = m_df["precision_mean"].to_numpy()
        rec_vals = m_df["recall_mean"].to_numpy()
        prauc_vals = m_df["pr_auc_mean"].to_numpy()

        summary_rows.append({
            "Model": m,
            "Features": n_feats,
            "Seeds": n_seeds,
            "Accuracy_Mean": np.mean(acc_vals),
            "Accuracy_SD": np.std(acc_vals, ddof=1) if n_seeds > 1 else 0.0,
            "Precision_Mean": np.mean(prec_vals),
            "Precision_SD": np.std(prec_vals, ddof=1) if n_seeds > 1 else 0.0,
            "Recall_Mean": np.mean(rec_vals),
            "Recall_SD": np.std(rec_vals, ddof=1) if n_seeds > 1 else 0.0,
            "F1_Mean": np.mean(f1_vals),
            "F1_SD": np.std(f1_vals, ddof=1) if n_seeds > 1 else 0.0,
            "PR_AUC_Mean": np.mean(prauc_vals),
            "PR_AUC_SD": np.std(prauc_vals, ddof=1) if n_seeds > 1 else 0.0,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "benchmark_m1_m7_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"[SAVED] Publication Summary Table: {summary_csv}")

    # Formatted display table
    display_rows = []
    for _, r in df_summary.iterrows():
        display_rows.append({
            "Model": r["Model"],
            "Features": r["Features"],
            "Accuracy": f"{r['Accuracy_Mean']*100:.2f}% (±{r['Accuracy_SD']*100:.2f}%)",
            "Precision": f"{r['Precision_Mean']*100:.2f}% (±{r['Precision_SD']*100:.2f}%)",
            "Recall": f"{r['Recall_Mean']*100:.2f}% (±{r['Recall_SD']*100:.2f}%)",
            "F1-Score": f"{r['F1_Mean']*100:.2f}% (±{r['F1_SD']*100:.2f}%)",
            "PR-AUC": f"{r['PR_AUC_Mean']:.4f} (±{r['PR_AUC_SD']:.4f})",
            "Seeds": r["Seeds"]
        })

    df_display = pd.DataFrame(display_rows)
    print("\n" + "="*85)
    print("MODEL COMPARISON BENCHMARK TABLE (M1 TO M7 ACROSS 5 SEEDS)")
    print("="*85)
    print(df_display.to_string(index=False))
    print("="*85 + "\n")

    # Generate LaTeX Table for Thesis
    latex_path = os.path.join(args.output_dir, "table_m1_m7_benchmark.tex")
    with open(latex_path, "w") as fp:
        fp.write("% LaTeX Table for Thesis: Model Comparison M1 to M7\n")
        fp.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        fp.write("\\caption{Model Specification Comparison: 5-Fold Stratified Group CV across 5 Random Seeds}\n")
        fp.write("\\label{tab:model_comparison}\n")
        fp.write("\\begin{tabular}{lcccccc}\n\\hline\\hline\n")
        fp.write("Model & Features & Accuracy & Precision & Recall & F1-Score & PR-AUC \\\\\n\\hline\n")
        for _, r in df_display.iterrows():
            fp.write(f"{r['Model']} & {r['Features']} & {r['Accuracy']} & {r['Precision']} & {r['Recall']} & {r['F1-Score']} & {r['PR-AUC']} \\\\\n")
        fp.write("\\hline\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"[SAVED] LaTeX Table saved to: {latex_path}")


if __name__ == "__main__":
    main()
