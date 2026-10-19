"""
tune_rf.py
Hyperparameter optimization engine using Optuna (Bayesian Optimization via TPE).
Evaluates parameter configurations using road-grouped cross-validation to prevent spatial overfitting.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import joblib

import optuna
from optuna.samplers import TPESampler

from src.config.config import PipelineConfig, PathConfig, HardwareConfig
from src.data.loader import (
    load_ground_truth_labels,
    load_harmonized_national_dataset,
    prepare_training_matrices
)

def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    road_ids: np.ndarray,
    feature_cols: List[str],
    n_jobs: int
) -> float:
    """
    Optuna objective function: evaluates hyperparameter trial using 3-fold GroupKFold.
    Maximizes Macro-F1 score to balance precision and recall.
    """
    # Sample hyperparameters from research-tailored search spaces
    n_estimators = trial.suggest_categorical("n_estimators", [200, 300, 500])
    max_depth = trial.suggest_int("max_depth", 15, 40)
    min_samples_split = trial.suggest_int("min_samples_split", 5, 30)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 16)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.3])
    criterion = trial.suggest_categorical("criterion", ["entropy", "gini"])

    # Fast 3-fold GroupKFold for hyperparameter exploration
    gkf = GroupKFold(n_splits=3)
    cv_f1_scores = []

    for train_idx, val_idx in gkf.split(X, y, groups=road_ids):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42 + trial.number
        )

        with joblib.parallel_backend("threading", n_jobs=n_jobs):
            rf.fit(X_tr, y_tr)

        val_preds = rf.predict(X_va)
        score = f1_score(y_va, val_preds, average="macro", zero_division=0)
        cv_f1_scores.append(score)

    return float(np.mean(cv_f1_scores))


def main():
    parser = argparse.ArgumentParser(description="FUCHS-CSC Random Forest Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="m5", choices=["m4", "m5", "m6", "m7"])
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna optimization trials.")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 20)))
    parser.add_argument("--features-dir", type=str, default="/beegfs/project/road_ml/data/parquet_national")
    parser.add_argument("--roads-shp", type=str,
                        default="/beegfs/project/road_ml/data/shapefiles/DistrictRoads_with training data.shp")
    parser.add_argument("--output-dir", type=str, default="/beegfs/project/road_ml/results/tuning")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ingest data
    config = PipelineConfig(model_name=args.model)
    feature_cols = config.get_features()

    labels_df = load_ground_truth_labels(Path(args.roads_shp))
    labeled_road_ids = set(labels_df["road_id"].unique())
    raw_data = load_harmonized_national_dataset(Path(args.features_dir), feature_cols, labeled_road_ids)
    X, y, road_ids, _ = prepare_training_matrices(raw_data, labels_df, feature_cols)

    study_name = f"rf_tuning_{args.model}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42)
    )

    print(f"\n[OPTUNA] Beginning Bayesian optimization sweep ({args.trials} trials) across {args.n_jobs} cores...")
    study.optimize(
        lambda trial: objective(trial, X, y, road_ids, feature_cols, args.n_jobs),
        n_trials=args.trials,
        timeout=14400  # 4-hour timeout safeguard
    )

    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print(f"Best Trial #{study.best_trial.number}")
    print(f"Best Macro-F1: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*70 + "\n")

    # Export results
    best_params_file = out_dir / f"best_params_{args.model}.json"
    with open(best_params_file, "w") as f:
        json.dump({"best_score": study.best_value, "params": study.best_params}, f, indent=2)

    df_trials = study.trials_dataframe()
    df_trials.to_csv(out_dir / f"trials_history_{args.model}.csv", index=False)
    print(f"[ARTIFACT] Tuning outputs written to: {best_params_file}")

if __name__ == "__main__":
    main()
