"""
train_rf.py
Primary Command-Line Interface (CLI) entry point for the FUCHS-CSC Random Forest training pipeline.
Supports Models M1–M7, 5-Fold GroupKFold CV, Slurm Job Arrays, and Full Production Fits.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import joblib
import pandas as pd

from src.config.config import PipelineConfig, PathConfig, RFHyperparameters, HardwareConfig
from src.data.loader import (
    load_ground_truth_labels,
    load_harmonized_national_dataset,
    prepare_training_matrices
)
from src.models.trainer import (
    train_and_evaluate_cv,
    train_single_fold,
    train_full_production_model
)

def parse_cli_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="FUCHS-CSC High-Performance Random Forest Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core Pipeline Arguments
    parser.add_argument("--model", type=str, choices=["m1", "m2", "m3", "m4", "m5", "m6", "m7"],
                        default="m5", help="Nested model family architecture to train.")
    parser.add_argument("--mode", type=str, choices=["cv", "full", "single_fold"],
                        default="cv", help="Execution mode: 5-fold CV, full fit, or single-fold array.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold index (0-4) when running in --mode single_fold.")
    parser.add_argument("--scope", type=str, choices=["national", "o_regions"],
                        default="national", help="Geographic scope.")

    # Storage Paths
    parser.add_argument("--features-dir", type=str, default="/beegfs/project/road_ml/data/parquet_national",
                        help="Path to directory containing regional Parquet files.")
    parser.add_argument("--roads-shp", type=str,
                        default="/beegfs/project/road_ml/data/shapefiles/DistrictRoads_with training data.shp",
                        help="Path to DistrictRoads shapefile with manual check labels.")
    parser.add_argument("--output-dir", type=str, default="/beegfs/project/road_ml/results",
                        help="Destination directory for serialized models, metrics, and logs.")
    parser.add_argument("--scratch-dir", type=str, default=os.environ.get("TMPDIR", "/localscratch"),
                        help="Fast node-local scratch storage path.")

    # Model Hyperparameters
    parser.add_argument("--n-trees", type=int, default=500, help="Number of trees in Random Forest.")
    parser.add_argument("--max-depth", type=int, default=25, help="Maximum tree depth (0 for unconstrained).")
    parser.add_argument("--min-samples-split", type=int, default=10, help="Min samples to split an internal node.")
    parser.add_argument("--min-samples-leaf", type=int, default=4, help="Min samples required at a leaf node.")
    parser.add_argument("--max-features", type=str, default="sqrt", help="Features per split: 'sqrt', 'log2'.")
    parser.add_argument("--criterion", type=str, choices=["entropy", "gini", "log_loss"], default="entropy",
                        help="Impurity split criterion.")
    parser.add_argument("--class-weight", type=str, default="balanced", help="Class weighting strategy.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")

    # Hardware & Concurrency
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 20)),
                        help="Number of CPU cores for parallel training (FUCHS-CSC: 20 cores).")
    parser.add_argument("--backend", type=str, choices=["threading", "loky"], default="threading",
                        help="Joblib parallelization backend.")

    args = parser.parse_args()

    # Build strongly-typed PipelineConfig
    paths = PathConfig(
        features_dir=Path(args.features_dir),
        roads_shp=Path(args.roads_shp),
        scratch_dir=Path(args.scratch_dir),
        output_dir=Path(args.output_dir)
    )

    max_depth = None if args.max_depth == 0 else args.max_depth
    rf_params = RFHyperparameters(
        n_estimators=args.n_trees,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        criterion=args.criterion,
        class_weight=args.class_weight,
        random_state=args.seed
    )

    hardware = HardwareConfig(
        n_jobs=args.n_jobs,
        joblib_backend=args.backend,
        joblib_temp_folder=Path(args.scratch_dir) / "joblib_cache"
    )

    return PipelineConfig(
        model_name=args.model,
        scope=args.scope,
        mode=args.mode,
        fold=args.fold,
        paths=paths,
        rf_params=rf_params,
        hardware=hardware
    )


def main():
    config = parse_cli_args()
    print(f"\n==============================================================================")
    print(f"FUCHS-CSC RANDOM FOREST PIPELINE EXECUTION START")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'LOCAL_INTERACTIVE')}")
    print(f"Model: {config.model_name.upper()} | Mode: {config.mode} | Cores: {config.hardware.n_jobs}")
    print(f"==============================================================================\n")

    # Ensure output directories exist
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.output_dir / "models").mkdir(exist_ok=True)
    (config.paths.output_dir / "metrics").mkdir(exist_ok=True)
    (config.paths.output_dir / "importance").mkdir(exist_ok=True)

    # 1. Parse Ground Truth Labels from Shapefile
    labels_df = load_ground_truth_labels(config.paths.roads_shp)
    labeled_road_ids = set(labels_df["road_id"].unique())
    print(f"[PRE-FLIGHT] Loaded {len(labels_df)} inspection records across {len(labeled_road_ids)} unique roads.")

    # 2. Ingest Regional Datasets under Master Schema Contract
    feature_cols = config.get_features()
    raw_data = load_harmonized_national_dataset(
        features_dir=config.paths.features_dir,
        feature_cols=feature_cols,
        labeled_road_ids=labeled_road_ids
    )

    # 3. Construct Training Matrices
    X, y, road_ids, labeled_df = prepare_training_matrices(raw_data, labels_df, feature_cols)

    # 4. Execute Mode
    if config.mode == "cv":
        results = train_and_evaluate_cv(X, y, road_ids, feature_cols, config)

        # Save Metrics
        metrics_file = config.paths.output_dir / "metrics" / f"metrics_{config.model_name}_{config.scope}.json"
        with open(metrics_file, "w") as f:
            json.dump(results["summary_stats"], f, indent=2)
        print(f"[ARTIFACT] Summary metrics written to: {metrics_file}")

        # Also fit final production model after CV
        model, importances, fit_time = train_full_production_model(X, y, feature_cols, config)
        model_file = config.paths.output_dir / "models" / f"RF_{config.model_name.upper()}_{config.scope}_full.joblib"
        joblib.dump({"model": model, "features": feature_cols, "config": vars(config.rf_params)},
                    model_file, compress=("zlib", 3))
        print(f"[ARTIFACT] Final production model serialized to: {model_file}")

        # Save Feature Importance
        imp_file = config.paths.output_dir / "importance" / f"importance_{config.model_name}_{config.scope}.csv"
        importances.to_csv(imp_file, header=["importance"])
        print(f"[ARTIFACT] Feature importances saved to: {imp_file}")

    elif config.mode == "single_fold":
        if config.fold is None:
            raise ValueError("--fold must be specified when using --mode single_fold")
        result = train_single_fold(X, y, road_ids, config.fold, feature_cols, config)

        fold_file = config.paths.output_dir / "metrics" / f"metrics_{config.model_name}_fold_{config.fold}.json"
        with open(fold_file, "w") as f:
            json.dump(result["metrics"], f, indent=2)

        model_file = config.paths.output_dir / "models" / f"RF_{config.model_name.upper()}_fold_{config.fold}.joblib"
        joblib.dump(result["model"], model_file, compress=("zlib", 3))
        print(f"[ARTIFACT] Fold {config.fold} model and metrics saved.")

    elif config.mode == "full":
        model, importances, fit_time = train_full_production_model(X, y, feature_cols, config)
        model_file = config.paths.output_dir / "models" / f"RF_{config.model_name.upper()}_{config.scope}_full.joblib"
        joblib.dump({"model": model, "features": feature_cols, "config": vars(config.rf_params)},
                    model_file, compress=("zlib", 3))
        print(f"[ARTIFACT] Production model saved to: {model_file}")

    print(f"\n==============================================================================")
    print(f"PIPELINE RUN COMPLETED SUCCESSFULLY: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"==============================================================================\n")

if __name__ == "__main__":
    main()
