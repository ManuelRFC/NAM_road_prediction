"""
spatial_transfer_audit.py
Spatial Transferability, Covariate Shift, and Out-of-Region Validation Audit.

Evaluates Model M5 segment predictions across Central and Southern Namibia:
1. Pure Control False Positive Rate (FPR) on unpaved roads (Earth/Gravel)
2. Historic Pre-1990 Paved Trunk Highway Stability (Always Paved TPR)
3. Regional Outlier Export for High-Resolution Satellite Verification
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any


def main():
    parser = argparse.ArgumentParser(description="Spatial Transferability & Control Stability Audit")
    parser.add_argument("--predictions-csv", type=str, required=True,
                        help="Path to namibia_national_segment_predictions_m5.csv")
    parser.add_argument("--timing-csv", type=str, default=None,
                        help="Path to road-level timing estimates CSV from changepoint_detector.py")
    parser.add_argument("--output-report", type=str, default="./spatial_transfer_audit_report.json",
                        help="Path to save summary audit metrics JSON")
    parser.add_argument("--outlier-csv", type=str, default="./candidate_breaks_audit.csv",
                        help="Path to save candidate transition roads for satellite inspection")
    args = parser.parse_args()

    print(f"[INPUT] Reading segment predictions: {args.predictions_csv}")
    df_pred = pd.read_csv(args.predictions_csv)
    df_pred["road_id"] = df_pred["road_id"].astype(str)

    # 1. Road-level aggregations
    road_stats = df_pred.groupby("road_id").agg(
        min_p=("mean_prob", "min"),
        max_p=("mean_prob", "max"),
        mean_p=("mean_prob", "mean"),
        q95_p=("mean_prob", lambda s: np.percentile(s, 95)),
        dyn_range=("mean_prob", lambda s: np.max(s) - np.min(s)),
        start_year=("start_year", "first"),
        end_year=("end_year", "first"),
        man_check=("man_check", "first"),
        SURFACE_TY=("SURFACE_TY", "first"),
        is_pure_control=("is_pure_control", "first")
    ).reset_index()

    # Merge timing estimates if available
    if args.timing_csv and os.path.exists(args.timing_csv):
        df_time = pd.read_csv(args.timing_csv)
        df_time["road_id"] = df_time["road_id"].astype(str)
        road_stats = road_stats.merge(
            df_time[["road_id", "has_transition", "pred_end_year", "r2"]],
            on="road_id", how="left"
        )
    else:
        road_stats["has_transition"] = (road_stats["dyn_range"] >= 0.35) & (road_stats["max_p"] >= 0.60)

    # 2. Pillar 1: Pure Controls Stability (Untreated Roads)
    pure_ctrl_mask = (
        (road_stats["man_check"].isin([4, 5])) |
        (road_stats["is_pure_control"] == True) |
        (road_stats["SURFACE_TY"].isin(["Earth Graded", "Earth Sand", "Earth Track", "Proclaimed Only"]))
    )
    df_ctrl = road_stats[pure_ctrl_mask]

    ctrl_metrics = {}
    if len(df_ctrl) > 0:
        false_positives = (df_ctrl["has_transition"] == True).sum()
        fpr = false_positives / len(df_ctrl)
        ctrl_metrics = {
            "n_pure_controls": int(len(df_ctrl)),
            "false_positive_count": int(false_positives),
            "false_positive_rate": float(fpr),
            "mean_control_probability": float(df_ctrl["mean_p"].mean()),
            "p95_control_probability": float(df_ctrl["q95_p"].mean()),
            "max_control_probability": float(df_ctrl["max_p"].max())
        }

    # 3. Pillar 2: Historic Always-Paved Anchors
    paved_mask = (road_stats["man_check"] == 9)
    df_paved = road_stats[paved_mask]

    paved_metrics = {}
    if len(df_paved) > 0:
        true_positives = (df_paved["mean_p"] >= 0.70).sum()
        tpr = true_positives / len(df_paved)
        paved_metrics = {
            "n_always_paved": int(len(df_paved)),
            "true_positive_count": int(true_positives),
            "true_positive_rate": float(tpr),
            "mean_paved_probability": float(df_paved["mean_p"].mean()),
            "min_paved_probability": float(df_paved["min_p"].min())
        }

    # 4. Pillar 3: Candidate Transitions for Satellite Audit
    # Roads that transitioned but were not in the verified training set
    candidate_mask = (
        road_stats["has_transition"] == True
    ) & (
        ~road_stats["man_check"].isin([1, 2, 6, 9])
    )
    df_candidates = road_stats[candidate_mask]
    if len(df_candidates) > 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.outlier_csv)), exist_ok=True)
        df_candidates.to_csv(args.outlier_csv, index=False)
        print(f"[AUDIT] Exported {len(df_candidates)} unverified candidate transition roads to: {args.outlier_csv}")

    report = {
        "total_roads_scored": int(len(road_stats)),
        "pure_control_audit": ctrl_metrics,
        "always_paved_audit": paved_metrics,
        "n_unverified_candidate_transitions": int(len(df_candidates))
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)

    print("\n==============================================================================")
    print("SPATIAL TRANSFERABILITY & OUT-OF-REGION AUDIT REPORT")
    print("==============================================================================")
    if ctrl_metrics:
        print(f"Pure Controls Evaluated:       {ctrl_metrics['n_pure_controls']}")
        print(f"Control False Positive Rate:   {ctrl_metrics['false_positive_rate']*100:.2f}% (Target: < 5.0%)")
        print(f"Mean Control Probability:      {ctrl_metrics['mean_control_probability']:.3f} (Target: < 0.20)")
        print(f"95th Percentile Probability:   {ctrl_metrics['p95_control_probability']:.3f}")
    if paved_metrics:
        print(f"Always-Paved Anchors:          {paved_metrics['n_always_paved']}")
        print(f"Always-Paved True Pos Rate:    {paved_metrics['true_positive_rate']*100:.2f}% (Target: >= 95.0%)")
    print(f"Unverified Candidate Breaks:   {report['n_unverified_candidate_transitions']}")
    print(f"Audit report saved:            {args.output_report}")
    print("==============================================================================\n")


if __name__ == "__main__":
    main()
