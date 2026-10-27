"""
changepoint_detector.py
Segment-Level Construction Timing Estimation and Econometric Metric Evaluator.

Implements the hybrid estimator:
1. 3-Year Rolling Median Filter
2. Dynamic Range Classification (Pure Control vs. Candidate Transition)
3. Bounded Parametric Logistic S-Curve Fitting
4. Segment-Level Econometric Error Metrics (MAE, MedAE, MBE, +/- 1 & 2 Year Accuracies, Spearman Rho)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from typing import Dict, Any, Optional, Tuple, List


def logistic_func(t, L, U, k, t0):
    """
    4-parameter bounded logistic curve:
    L: lower baseline probability (pre-construction dirt/gravel)
    U: upper asymptote probability (post-construction bitumen/paved)
    k: transition steepness / rate
    t0: inflection year (midpoint of construction)
    """
    return L + (U - L) / (1.0 + np.exp(-np.clip(k * (t - t0), -50, 50)))


def fit_logistic_transition(
    years: np.ndarray,
    probs: np.ndarray
) -> Tuple[Optional[Dict[str, float]], float]:
    """
    Fits bounded logistic S-curve to annual probability series.
    Returns (params_dict, r_squared).
    """
    if len(years) < 4:
        return None, 0.0

    # Smooth input probabilities with 3-year rolling median
    s_probs = pd.Series(probs).rolling(window=3, min_periods=1, center=True).median().to_numpy()

    p_min = np.min(s_probs)
    p_max = np.max(s_probs)
    dyn_range = p_max - p_min

    # If dynamic range is too small, road never experienced paving transition
    if dyn_range < 0.30 and p_max < 0.40:
        return None, 0.0

    # Initial parameter guesses
    t_mid_guess = years[np.argmax(np.gradient(s_probs))]
    p0 = [max(0.0, p_min), min(1.0, p_max), 1.0, t_mid_guess]
    bounds = (
        [0.0, 0.50, 0.1, years[0] - 2],    # Lower bounds [L, U, k, t0]
        [0.40, 1.0, 5.0, years[-1] + 2]     # Upper bounds
    )

    try:
        popt, _ = curve_fit(logistic_func, years, s_probs, p0=p0, bounds=bounds, maxfev=2000)
        L_fit, U_fit, k_fit, t0_fit = popt

        # Calculate R^2 goodness of fit
        residuals = s_probs - logistic_func(years, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((s_probs - np.mean(s_probs)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Physical transition timing
        # Half-transition width = 1.5 / k
        delta_t = 1.5 / max(k_fit, 0.1)
        start_year = float(np.clip(t0_fit - delta_t, years[0], years[-1]))
        end_year = float(np.clip(t0_fit + delta_t, years[0], years[-1]))

        result = {
            "L": float(L_fit),
            "U": float(U_fit),
            "k": float(k_fit),
            "t0": float(t0_fit),
            "pred_start_year": start_year,
            "pred_mid_year": float(t0_fit),
            "pred_end_year": end_year,
            "r2": float(r2),
            "dynamic_range": float(U_fit - L_fit)
        }
        return result, r2

    except Exception:
        # Fallback heuristic: threshold crossing (0.15 / 0.75)
        above_start = years[s_probs >= 0.15]
        above_end = years[s_probs >= 0.75]

        start_yr = float(above_start[0]) if len(above_start) > 0 else np.nan
        end_yr = float(above_end[0]) if len(above_end) > 0 else start_yr

        if not np.isnan(start_yr):
            return {
                "L": float(p_min),
                "U": float(p_max),
                "k": 1.0,
                "t0": float((start_yr + end_yr) / 2.0),
                "pred_start_year": start_yr,
                "pred_mid_year": float((start_yr + end_yr) / 2.0),
                "pred_end_year": end_yr,
                "r2": 0.50,
                "dynamic_range": float(dyn_range)
            }, 0.50

        return None, 0.0


def evaluate_segment_timing(df_predictions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs changepoint detection per road and computes evaluation metrics against ground truth.
    """
    roads = df_predictions["road_id"].unique()
    results = []

    for rid in roads:
        sub = df_predictions[df_predictions["road_id"] == rid].sort_values("year").copy()
        years = sub["year"].to_numpy()
        probs = sub["mean_prob"].to_numpy()

        fit_res, r2 = fit_logistic_transition(years, probs)

        row = {
            "road_id": rid,
            "man_check": sub["man_check"].iloc[0] if "man_check" in sub.columns else np.nan,
            "SURFACE_TY": sub["SURFACE_TY"].iloc[0] if "SURFACE_TY" in sub.columns else "Unknown",
            "true_start_year": sub["start_year"].iloc[0] if "start_year" in sub.columns else np.nan,
            "true_end_year": sub["end_year"].iloc[0] if "end_year" in sub.columns else np.nan,
            "has_transition": fit_res is not None,
            "r2": r2 if fit_res is not None else 0.0,
            "pred_start_year": fit_res["pred_start_year"] if fit_res else np.nan,
            "pred_mid_year": fit_res["pred_mid_year"] if fit_res else np.nan,
            "pred_end_year": fit_res["pred_end_year"] if fit_res else np.nan,
            "mean_prob_1990_1995": probs[:5].mean() if len(probs) >= 5 else np.nan,
            "mean_prob_2019_2024": probs[-5:].mean() if len(probs) >= 5 else np.nan,
        }
        results.append(row)

    eval_df = pd.DataFrame(results)

    # Filter to ground-truth treated roads with verified completion years
    eval_treated = eval_df[
        eval_df["true_end_year"].notna() &
        eval_df["pred_end_year"].notna() &
        eval_df["man_check"].isin([1, 2, 6])
    ].copy()

    metrics = {}
    if len(eval_treated) > 0:
        y_true = eval_treated["true_end_year"].to_numpy()
        y_pred = eval_treated["pred_end_year"].to_numpy()
        diff = y_pred - y_true
        abs_diff = np.abs(diff)

        rho, p_val = spearmanr(y_true, y_pred)

        metrics = {
            "n_evaluated_roads": len(eval_treated),
            "mean_absolute_error_years": float(np.mean(abs_diff)),
            "median_absolute_error_years": float(np.median(abs_diff)),
            "mean_bias_error_years": float(np.mean(diff)),
            "root_mean_squared_error": float(np.sqrt(np.mean(diff ** 2))),
            "exact_match_accuracy": float(np.mean(abs_diff == 0)),
            "within_1_year_accuracy": float(np.mean(abs_diff <= 1.0)),
            "within_2_years_accuracy": float(np.mean(abs_diff <= 2.0)),
            "within_3_years_accuracy": float(np.mean(abs_diff <= 3.0)),
            "spearman_rank_correlation": float(rho),
            "spearman_p_value": float(p_val)
        }

    return eval_df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True,
                        help="Path to namibia_national_segment_predictions_m5.csv")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Path to save road-level estimated timing CSV")
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="Path to save summary evaluation metrics JSON")
    args = parser.parse_args()

    print(f"[INPUT] Reading segment predictions from: {args.input_csv}")
    df_pred = pd.read_csv(args.input_csv)
    print(f"[INPUT] Loaded {len(df_pred):,} rows across {df_pred['road_id'].nunique():,} roads.")

    eval_df, metrics = evaluate_segment_timing(df_pred)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    eval_df.to_csv(args.output_csv, index=False)
    print(f"[OUTPUT] Saved road-level timing estimates to: {args.output_csv}")

    if metrics:
        print("\n==============================================================================")
        print("SEGMENT-LEVEL CONSTRUCTION TIMING EVALUATION SUMMARY")
        print("==============================================================================")
        print(f"Evaluated Roads with Ground Truth: {metrics['n_evaluated_roads']}")
        print(f"Mean Absolute Error (MAE):         {metrics['mean_absolute_error_years']:.2f} years")
        print(f"Median Absolute Error (MedAE):     {metrics['median_absolute_error_years']:.2f} years")
        print(f"Mean Bias Error (MBE):             {metrics['mean_bias_error_years']:+.2f} years")
        print(f"Root Mean Squared Error (RMSE):    {metrics['root_mean_squared_error']:.2f} years")
        print(f"Exact Year Match:                  {metrics['exact_match_accuracy']*100:.1f}%")
        print(f"Within +/- 1 Year Accuracy:        {metrics['within_1_year_accuracy']*100:.1f}%")
        print(f"Within +/- 2 Years Accuracy:       {metrics['within_2_years_accuracy']*100:.1f}%")
        print(f"Spearman Rank Correlation:         {metrics['spearman_rank_correlation']:.3f} (p={metrics['spearman_p_value']:.2e})")
        print("==============================================================================\n")

        if args.metrics_json:
            import json
            with open(args.metrics_json, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"[OUTPUT] Metrics saved to: {args.metrics_json}")


if __name__ == "__main__":
    main()
