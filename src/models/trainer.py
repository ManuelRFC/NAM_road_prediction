"""
src/models/trainer.py
High-performance Random Forest training engine for FUCHS-CSC cluster.
Implements leak-free GroupKFold cross-validation, 20-core joblib threading/loky,
and comprehensive out-of-fold prediction tracking.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import joblib

from src.config.config import PipelineConfig, RFHyperparameters
from src.evaluation.metrics import compute_classification_metrics

def build_rf_classifier(params: RFHyperparameters, n_jobs: int) -> RandomForestClassifier:
    """Instantiates a Scikit-Learn RandomForestClassifier with FUCHS-CSC parameters."""
    return RandomForestClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        max_features=params.max_features,
        criterion=params.criterion,
        class_weight=params.class_weight,
        bootstrap=params.bootstrap,
        n_jobs=n_jobs,
        random_state=params.random_state
    )

def train_and_evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    road_ids: np.ndarray,
    feature_names: List[str],
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Executes leak-free 5-Fold GroupKFold Cross-Validation on FUCHS-CSC.
    Guarantees no spatial data leakage: all observations from a road belong
    strictly to either training folds or the validation fold.
    """
    gkf = GroupKFold(n_splits=config.k_folds)
    n_samples = len(y)

    oof_predictions = np.zeros(n_samples, dtype=np.int8)
    oof_probabilities = np.zeros(n_samples, dtype=np.float32)
    fold_metrics: List[Dict[str, float]] = []

    print(f"\n{'='*70}")
    print(f"STARTING {config.k_folds}-FOLD GROUP-KFOLD CROSS-VALIDATION")
    print(f"Model: {config.model_name.upper()} | Features: {len(feature_names)} | Samples: {n_samples:,} | Roads: {len(np.unique(road_ids))}")
    print(f"Hardware: {config.hardware.n_jobs} Cores | Backend: {config.hardware.joblib_backend}")
    print(f"{'='*70}\n")

    start_total = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=road_ids)):
        fold_num = fold_idx + 1
        print(f"--- FOLD {fold_num}/{config.k_folds} ---")
        t0 = time.time()

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_roads = len(np.unique(road_ids[train_idx]))
        val_roads = len(np.unique(road_ids[val_idx]))
        print(f"Train: {len(y_train):,} samples ({train_roads} roads) | Val: {len(y_val):,} samples ({val_roads} roads)")

        # Create model with fold-specific random seed
        fold_params = RFHyperparameters(**vars(config.rf_params))
        fold_params.random_state = config.rf_params.random_state + fold_idx
        rf = build_rf_classifier(fold_params, n_jobs=config.hardware.n_jobs)

        # Train model using configured Joblib backend
        with joblib.parallel_backend(config.hardware.joblib_backend, n_jobs=config.hardware.n_jobs):
            rf.fit(X_train, y_train)

        fit_time = time.time() - t0

        # Predict validation probabilities and class labels
        val_probs = rf.predict_proba(X_val)[:, 1].astype(np.float32)
        val_preds = (val_probs >= 0.5).astype(np.int8)

        oof_probabilities[val_idx] = val_probs
        oof_predictions[val_idx] = val_preds

        # Evaluate fold metrics
        metrics = compute_classification_metrics(y_val, val_preds, val_probs)
        metrics["fit_time_sec"] = fit_time
        fold_metrics.append(metrics)

        print(f"Fold {fold_num} Completed in {fit_time:.1f}s | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}\n")

    total_cv_time = time.time() - start_total

    # Compute overall Out-Of-Fold (OOF) metrics
    overall_oof_metrics = compute_classification_metrics(y, oof_predictions, oof_probabilities)
    overall_oof_metrics["total_cv_time_sec"] = total_cv_time

    # Calculate mean and standard deviation across folds
    summary_stats = {}
    for metric_key in ["accuracy", "precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc"]:
        values = [m[metric_key] for m in fold_metrics]
        summary_stats[f"{metric_key}_mean"] = float(np.mean(values))
        summary_stats[f"{metric_key}_std"] = float(np.std(values))

    print(f"{'='*70}")
    print(f"CROSS-VALIDATION RESULTS SUMMARY: {config.model_name.upper()}")
    print(f"Mean Accuracy:  {summary_stats['accuracy_mean']:.4f} +/- {summary_stats['accuracy_std']:.4f}")
    print(f"Mean Precision: {summary_stats['precision_mean']:.4f} +/- {summary_stats['precision_std']:.4f}")
    print(f"Mean Recall:    {summary_stats['recall_mean']:.4f} +/- {summary_stats['recall_std']:.4f}")
    print(f"Mean F1-Score:  {summary_stats['f1_mean']:.4f} +/- {summary_stats['f1_std']:.4f}")
    print(f"Mean PR-AUC:    {summary_stats['pr_auc_mean']:.4f} +/- {summary_stats['pr_auc_std']:.4f}")
    print(f"Total Execution Time: {total_cv_time / 60:.1f} minutes")
    print(f"{'='*70}\n")

    return {
        "fold_metrics": fold_metrics,
        "summary_stats": summary_stats,
        "overall_oof_metrics": overall_oof_metrics,
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities
    }


def train_single_fold(
    X: np.ndarray,
    y: np.ndarray,
    road_ids: np.ndarray,
    target_fold: int,
    feature_names: List[str],
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Trains a single fold in isolation (for Slurm Job Array execution: --array=0-4).
    """
    gkf = GroupKFold(n_splits=config.k_folds)
    splits = list(gkf.split(X, y, groups=road_ids))

    if target_fold < 0 or target_fold >= config.k_folds:
        raise ValueError(f"target_fold must be between 0 and {config.k_folds - 1}, got {target_fold}")

    train_idx, val_idx = splits[target_fold]
    print(f"[SLURM ARRAY TASK] Executing Fold {target_fold} (1 of {config.k_folds})")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    t0 = time.time()
    rf = build_rf_classifier(config.rf_params, n_jobs=config.hardware.n_jobs)

    with joblib.parallel_backend(config.hardware.joblib_backend, n_jobs=config.hardware.n_jobs):
        rf.fit(X_train, y_train)

    fit_time = time.time() - t0
    val_probs = rf.predict_proba(X_val)[:, 1].astype(np.float32)
    val_preds = (val_probs >= 0.5).astype(np.int8)

    metrics = compute_classification_metrics(y_val, val_preds, val_probs)
    metrics["fit_time_sec"] = fit_time
    metrics["fold"] = target_fold

    # Compute Feature Importances
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    return {
        "fold": target_fold,
        "model": rf,
        "metrics": metrics,
        "val_indices": val_idx,
        "val_preds": val_preds,
        "val_probs": val_probs,
        "importances": importances
    }


def train_full_production_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: PipelineConfig
) -> Tuple[RandomForestClassifier, pd.Series, float]:
    """
    Trains the final production Random Forest model on 100% of available labeled data.
    """
    print(f"\n[PRODUCTION FIT] Training full model on all {len(y):,} labeled rows using {config.hardware.n_jobs} cores...")
    t0 = time.time()

    rf = build_rf_classifier(config.rf_params, n_jobs=config.hardware.n_jobs)
    with joblib.parallel_backend(config.hardware.joblib_backend, n_jobs=config.hardware.n_jobs):
        rf.fit(X, y)

    elapsed = time.time() - t0
    print(f"[PRODUCTION FIT] Completed 500-tree ensemble in {elapsed:.1f}s ({elapsed / 60:.2f} mins).")

    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    return rf, importances, elapsed
