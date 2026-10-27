"""
train_learning_curve.py
HPC Sample Complexity / Learning Curve Experiment Worker for FUCHS-CSC.

Evaluates how model performance scales with the quantity of training roads.
Subsamples road segments (to prevent spatial autocorrelation leakage) across
fractions [0.05, 0.10, 0.25, 0.50, 0.75, 1.00] and evaluates on a fixed 20%
spatial holdout test set of untouched roads.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

from src.config.config import M5_FEATURES
from src.data.loader import (
    load_ground_truth_labels,
    load_harmonized_national_dataset,
    prepare_training_matrices
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FUCHS-CSC Learning Curve Experiment Task Worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fraction", type=float, required=True,
                        help="Proportion of training pool roads to sample (0.05 to 1.00)")
    parser.add_argument("--n-reps", type=int, default=5,
                        help="Number of Monte Carlo subsampling repetitions")
    parser.add_argument("--features-dir", type=str, required=True,
                        help="Path to directory containing regional Parquet feature files")
    parser.add_argument("--roads-shp", type=str, required=True,
                        help="Path to DistrictRoads shapefile")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save task metrics CSV")
    parser.add_argument("--cache-file", type=str, default=None,
                        help="Path to cached parquet file containing all labeled rows")
    parser.add_argument("--n-trees", type=int, default=100,
                        help="Number of trees in ensemble (100 for fast convergence across 26 runs)")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 20)),
                        help="CPU cores for parallel tree fitting")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed for fixed test-train split")
    return parser.parse_args()


def get_or_create_labeled_dataset(args, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads labeled training data either from fast local scratch parquet cache or raw Parquet files.
    """
    labels_df = load_ground_truth_labels(Path(args.roads_shp))
    labeled_road_ids = set(labels_df["road_id"].unique())

    if args.cache_file and os.path.exists(args.cache_file):
        print(f"[CACHE] Loading pre-processed labeled data from: {args.cache_file}")
        t0 = time.time()
        labeled_df = pd.read_parquet(args.cache_file)
        print(f"[CACHE] Loaded {len(labeled_df):,} rows across {labeled_df['road_id'].nunique():,} roads in {time.time() - t0:.2f}s.")
        return labeled_df, labels_df

    print(f"[DATA LOADER] Ingesting raw Parquet files for {len(labeled_road_ids)} labeled roads...")
    raw_data = load_harmonized_national_dataset(
        features_dir=Path(args.features_dir),
        feature_cols=feature_cols,
        labeled_road_ids=labeled_road_ids
    )

    X, y, road_ids, labeled_df = prepare_training_matrices(raw_data, labels_df, feature_cols)

    if args.cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache_file)), exist_ok=True)
        print(f"[CACHE] Saving labeled dataset to fast scratch cache: {args.cache_file}")
        labeled_df.to_parquet(args.cache_file, index=False)

    return labeled_df, labels_df


def main():
    args = parse_args()
    start_total = time.time()

    print("==============================================================================")
    print("FUCHS-CSC LEARNING CURVE TASK WORKER")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"Fraction: {args.fraction:.2f} | Repetitions: {args.n_reps}")
    print(f"Cores: {args.n_jobs} | Trees per RF: {args.n_trees}")
    print(f"Output Directory: {args.output_dir}")
    print("==============================================================================\n")

    os.makedirs(args.output_dir, exist_ok=True)
    feature_cols = M5_FEATURES

    # 1. Load or Ingest Labeled Dataset
    labeled_df, labels_df = get_or_create_labeled_dataset(args, feature_cols)
    total_roads = sorted(labeled_df["road_id"].unique())
    print(f"[DATASET] Total available labeled roads: {len(total_roads)}")

    # 2. Fixed 80/20 Spatial Road Holdout (Deterministic across all array tasks)
    rng_master = np.random.default_rng(args.seed)
    shuffled_roads = rng_master.permutation(total_roads)

    n_test_roads = max(10, int(np.round(0.20 * len(total_roads))))
    test_roads = set(shuffled_roads[:n_test_roads])
    train_pool_roads = np.array(shuffled_roads[n_test_roads:])

    print(f"[SPLIT] Fixed Holdout Test Set: {len(test_roads)} roads")
    print(f"[SPLIT] Training Pool Set: {len(train_pool_roads)} roads")

    # Construct Fixed Test Matrix
    test_mask = labeled_df["road_id"].isin(test_roads)
    df_test = labeled_df[test_mask]
    X_test = df_test[feature_cols].to_numpy(dtype=np.float32)
    y_test = (df_test["label"] == "Post_Construction").astype(np.int8).to_numpy()

    # Pre-filter training pool dataframe
    df_train_pool = labeled_df[~test_mask]
    print(f"[TEST] Held-out test matrix: {len(X_test):,} samples (Pos: {(y_test==1).sum():,}, Neg: {(y_test==0).sum():,})")

    # 3. Determine Number of Roads for this Fraction
    n_pool = len(train_pool_roads)
    if args.fraction >= 1.0:
        k_roads = n_pool
        reps = 1  # Full pool is deterministic
    else:
        k_roads = max(3, int(np.round(args.fraction * n_pool)))
        reps = args.n_reps

    records: List[Dict[str, Any]] = []

    for rep in range(1, reps + 1):
        rep_seed = int(args.seed + args.fraction * 10000 + rep * 101)
        rng_rep = np.random.default_rng(rep_seed)

        if args.fraction >= 1.0:
            sampled_roads = train_pool_roads
        else:
            sampled_roads = rng_rep.choice(train_pool_roads, size=k_roads, replace=False)

        # Slice training subset by sampled roads
        train_sub_mask = df_train_pool["road_id"].isin(set(sampled_roads))
        df_train_sub = df_train_pool[train_sub_mask]

        X_train = df_train_sub[feature_cols].to_numpy(dtype=np.float32)
        y_train = (df_train_sub["label"] == "Post_Construction").astype(np.int8).to_numpy()

        print(f"\n--- [RUN {rep}/{reps}] Fraction: {args.fraction:.2f} | Roads: {len(sampled_roads)} | Pixels: {len(X_train):,} ---")

        t_fit_start = time.time()
        rf = RandomForestClassifier(
            n_estimators=args.n_trees,
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            criterion="entropy",
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=rep_seed
        )
        rf.fit(X_train, y_train)
        fit_duration = time.time() - t_fit_start

        # Predict on untouched test set
        preds = rf.predict(X_test)
        probs = rf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        pr_auc = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)

        print(f"Results: Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} (Fit: {fit_duration:.1f}s)")

        records.append({
            "fraction": args.fraction,
            "repetition": rep,
            "seed": rep_seed,
            "n_roads": len(sampled_roads),
            "n_pixels": len(X_train),
            "test_roads": len(test_roads),
            "test_pixels": len(X_test),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "fit_time_sec": fit_duration
        })

    # 4. Save Task Output CSV
    task_csv = os.path.join(args.output_dir, f"learning_curve_fraction_{args.fraction:.2f}.csv")
    df_results = pd.DataFrame(records)
    df_results.to_csv(task_csv, index=False)
    print(f"\n[ARTIFACT] Task results written to: {task_csv}")
    print(f"[TOTAL TIME] Completed fraction {args.fraction:.2f} in {(time.time() - start_total)/60:.2f} mins.")


if __name__ == "__main__":
    main()
