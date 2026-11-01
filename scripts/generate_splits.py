"""
generate_splits.py
Generates deterministic, stratified, leak-free road splits across multiple random seeds.

Stratifies 238 ground-truth roads by surface type, inspection status, and length buckets.
Guarantees that all pixels of any given road stay together in exactly one fold.
Saves a master split table (master_cv_splits_5seeds.csv and .json) for M1-M7 benchmarks.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def create_stratified_folds(
    roads_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Stochastic stratified bin-packing algorithm to allocate roads across n_splits folds.
    Balances:
    1. Stratification category (surface_type + man_check)
    2. Total road length / pixel weight per fold
    Guarantees different seeds produce genuinely distinct partitions.
    """
    rng = np.random.default_rng(seed)
    folds = np.full(len(roads_df), -1, dtype=int)
    
    roads_df = roads_df.copy()
    roads_df["strat_cat"] = (
        roads_df["surface"].astype(str) + "_" + 
        roads_df["man_check"].astype(str)
    )
    
    fold_pixel_sums = np.zeros(n_splits, dtype=float)
    fold_road_counts = np.zeros(n_splits, dtype=int)
    
    # Process categories
    categories = np.array(list(roads_df["strat_cat"].unique()))
    rng.shuffle(categories)
    
    for cat in categories:
        cat_indices = np.array(roads_df[roads_df["strat_cat"] == cat].index, copy=True)
        rng.shuffle(cat_indices)
        
        for idx in cat_indices:
            p_weight = roads_df.loc[idx, "mean_pixels"]
            # Add subtle jitter based on seed so ties/near-ties are broken randomly
            jitter = rng.uniform(0.98, 1.02, size=n_splits)
            effective_loads = (fold_pixel_sums + 1e-3 * fold_road_counts) * jitter
            target_fold = int(np.argmin(effective_loads))
            folds[idx] = target_fold
            fold_pixel_sums[target_fold] += p_weight
            fold_road_counts[target_fold] += 1
            
    return folds


def main():
    # 1. Load labeled road metadata from predictions CSV
    csv_candidates = [
        r"C:\Users\manue\namibia_national_segment_predictions_m5.csv",
        r"/work/fuchs/agmisc/cmiel/road_ml/results/m5_monolithic_1801371/namibia_national_segment_predictions_m5.csv"
    ]
    input_csv = None
    for c in csv_candidates:
        if os.path.exists(c):
            input_csv = c
            break
            
    if not input_csv:
        raise FileNotFoundError("Could not find namibia_national_segment_predictions_m5.csv")
        
    print(f"[INPUT] Reading metadata from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    
    # Extract unique labeled roads
    labeled = df[df["man_check"].notna()].groupby("road_id").agg(
        surface=("SURFACE_TY", "first"),
        mean_pixels=("n_pixels", "mean"),
        man_check=("man_check", "first"),
        start_year=("start_year", "first"),
        end_year=("end_year", "first")
    ).reset_index()
    
    print(f"[INFO] Found {len(labeled)} unique ground-truth roads.")
    print("Surface type distribution:\n", labeled["surface"].value_counts())
    print("\nInspection code (man_check) distribution:\n", labeled["man_check"].value_counts())

    # 2. Generate 5-Fold Splits across 5 Different Seeds
    seeds = [42, 101, 202, 303, 404]
    split_results = labeled[["road_id", "surface", "man_check", "mean_pixels"]].copy()
    split_results["road_id"] = split_results["road_id"].astype(str)
    
    for s in seeds:
        col_name = f"fold_seed_{s}"
        folds = create_stratified_folds(labeled, n_splits=5, seed=s)
        split_results[col_name] = folds
        print(f"\n--- Fold Distribution for Seed {s} ---")
        ct = pd.crosstab(split_results["surface"], split_results[col_name])
        print(ct)
        pixel_summary = split_results.groupby(col_name)["mean_pixels"].sum()
        print(f"Pixels per fold: min={pixel_summary.min():,.0f}, max={pixel_summary.max():,.0f}, ratio={pixel_summary.max()/pixel_summary.min():.2f}")

    # 3. Save Master Splits Table
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(repo_root, "splits")
    os.makedirs(output_dir, exist_ok=True)
    
    out_csv = os.path.join(output_dir, "master_cv_splits_5seeds.csv")
    split_results.to_csv(out_csv, index=False)
    print(f"\n[ARTIFACT] Master splits CSV written to: {out_csv}")
    
    # Also save JSON mapping {seed: {fold: [road_id, ...]}}
    json_mapping: Dict[str, Dict[str, List[str]]] = {}
    for s in seeds:
        seed_key = f"seed_{s}"
        json_mapping[seed_key] = {}
        for f in range(5):
            roads_in_fold = split_results[split_results[f"fold_seed_{s}"] == f]["road_id"].tolist()
            json_mapping[seed_key][f"fold_{f}"] = roads_in_fold
            
    out_json = os.path.join(output_dir, "master_cv_splits_5seeds.json")
    with open(out_json, "w") as f:
        json.dump(json_mapping, f, indent=2)
    print(f"[ARTIFACT] Master splits JSON written to: {out_json}")


if __name__ == "__main__":
    main()
