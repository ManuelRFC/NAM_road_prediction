"""
predict_segments.py
HPC Batch Streaming Inference and Segment-Level Aggregation for FUCHS-CSC.

Loads Model M5, streams national Parquet feature files in memory-mapped batches,
evaluates post-construction probabilities, aggregates to the road-segment-year level,
and merges ground-truth inspection windows from the Roads Authority shapefile.
"""

import argparse
import gc
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config.config import M5_FEATURES, METADATA_COLUMNS
from src.data.loader import load_ground_truth_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="FUCHS-CSC Streaming Segment-Level Inference & Aggregation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to serialized Random Forest checkpoint (.joblib)")
    parser.add_argument("--features-dir", type=str, required=True,
                        help="Path to directory containing regional Parquet feature files")
    parser.add_argument("--roads-shp", type=str, default=None,
                        help="Path to DistrictRoads shapefile for ground-truth metadata annotation")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Path to save segment-level aggregated CSV table")
    parser.add_argument("--output-parquet", type=str, default=None,
                        help="Optional path to save segment-level aggregated Parquet table")
    parser.add_argument("--batch-size", type=int, default=250_000,
                        help="Rows per streaming PyArrow record batch")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 20)),
                        help="Parallel CPU cores for tree evaluation")
    parser.add_argument("--only-labeled-roads", action="store_true",
                        help="If set, only score roads present in the ground-truth shapefile")
    return parser.parse_args()


def main():
    args = parse_args()
    start_total = time.time()

    print("==============================================================================")
    print("FUCHS-CSC SEGMENT INFERENCE & AGGREGATION PIPELINE")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"Node: {os.uname().nodename if hasattr(os, 'uname') else 'LOCAL'}")
    print(f"Model Checkpoint: {args.model_path}")
    print(f"Features Directory: {args.features_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Allocated Cores: {args.n_jobs} | Batch Size: {args.batch_size:,} rows")
    print("==============================================================================\n")

    # 1. Load Ground Truth Labels Metadata (if provided)
    labels_meta = None
    target_road_ids: Optional[Set[str]] = None
    if args.roads_shp and os.path.exists(args.roads_shp):
        print(f"[PRE-FLIGHT] Loading road metadata from: {args.roads_shp}")
        labels_meta = load_ground_truth_labels(Path(args.roads_shp))
        target_road_ids = set(labels_meta["road_id"].unique())
        print(f"[PRE-FLIGHT] Found {len(labels_meta)} inspection records across {len(target_road_ids)} unique roads.")
        if args.only_labeled_roads:
            print(f"[FILTER] Restricting inference strictly to {len(target_road_ids)} labeled roads.")
    else:
        print("[PRE-FLIGHT] No shapefile provided or found. Scoring all roads in Parquet files.")

    # 2. Load Model with Memory Mapping
    print(f"[MODEL] Loading model checkpoint from: {args.model_path}")
    t0 = time.time()
    loaded = joblib.load(args.model_path, mmap_mode="r")
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        feature_cols = loaded.get("features", getattr(model, "feature_names_in_", None))
    else:
        model = loaded
        feature_cols = getattr(model, "feature_names_in_", None)

    if feature_cols is None:
        print("[WARN] Model does not contain explicit feature names. Falling back to M5_FEATURES specification.")
        feature_cols = M5_FEATURES
    else:
        feature_cols = list(feature_cols)

    # Configure worker threads
    if hasattr(model, "n_jobs"):
        model.n_jobs = args.n_jobs

    print(f"[MODEL] Loaded model successfully in {time.time() - t0:.2f}s. Enforcing {len(feature_cols)} features.")

    # 3. Locate Feature Files
    parquet_files = sorted(glob.glob(os.path.join(args.features_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in directory: {args.features_dir}")
    print(f"[DATA] Located {len(parquet_files)} regional Parquet feature files.")

    # We need road_id, year, pixel_type (if present), and all model feature columns
    chunk_accumulators: List[pd.DataFrame] = []
    total_processed_rows = 0
    total_scored_rows = 0

    t_infer_start = time.time()

    for f_idx, filepath in enumerate(parquet_files, 1):
        file_basename = os.path.basename(filepath)
        print(f"\n--- [FILE {f_idx:02d}/{len(parquet_files):02d}] Ingesting {file_basename} ---")

        pf = pq.ParquetFile(filepath)
        existing_cols = set(pf.schema.names)

        # Build column projection
        meta_cols = [c for c in ["road_id", "year", "pixel_type", "pixel_type_new"] if c in existing_cols]
        active_features_in_file = [c for c in feature_cols if c in existing_cols]
        cols_to_read = list(set(meta_cols + active_features_in_file))

        missing_in_file = [c for c in feature_cols if c not in existing_cols]
        if missing_in_file:
            print(f"[SCHEMA] Notice: {len(missing_in_file)} features missing in {file_basename}. Will zero-impute.")

        file_rows = pf.metadata.num_rows
        file_scored = 0

        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=args.batch_size, columns=cols_to_read), 1):
            df_chunk = batch.to_pandas()
            n_raw = len(df_chunk)
            total_processed_rows += n_raw

            # 1. Normalize road_id
            df_chunk["road_id"] = df_chunk["road_id"].astype(str)

            # 2. Filter to labeled roads if requested
            if target_road_ids is not None and args.only_labeled_roads:
                df_chunk = df_chunk[df_chunk["road_id"].isin(target_road_ids)]
                if len(df_chunk) == 0:
                    continue

            # 3. Filter to road pixels only (pixel_type == 'on_road')
            for p_col in ["pixel_type", "pixel_type_new"]:
                if p_col in df_chunk.columns:
                    df_chunk = df_chunk[df_chunk[p_col] == "on_road"]
                    break

            if len(df_chunk) == 0:
                continue

            # 4. Impute missing schema features with float32 0.0
            for missing_col in missing_in_file:
                df_chunk[missing_col] = np.float32(0.0)

            # 5. Extract Feature Matrix strictly aligned to feature_cols
            X_chunk = df_chunk[feature_cols].to_numpy(dtype=np.float32)

            # Replace any stray NaNs or Infs with 0.0
            np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # 6. Predict Post-Construction Probability (Class 1)
            # Binary classifier: column 1 is P(Post_Construction)
            probs = model.predict_proba(X_chunk)[:, 1].astype(np.float32)

            df_chunk["prob_post"] = probs
            file_scored += len(probs)
            total_scored_rows += len(probs)

            # 7. Intermediate Chunk Aggregation to immediately compress RAM
            chunk_agg = df_chunk.groupby(["road_id", "year"]).agg(
                sum_prob=("prob_post", "sum"),
                sum_sq_prob=("prob_post", lambda s: np.sum(s**2)),
                n_pixels=("prob_post", "count"),
                median_prob=("prob_post", "median")
            ).reset_index()

            chunk_accumulators.append(chunk_agg)
            del df_chunk, X_chunk, probs

        gc.collect()
        print(f"[FILE {f_idx:02d}] Scored {file_scored:,} / {file_rows:,} pixels.")

    print(f"\n==============================================================================")
    print(f"BATCH SCORING COMPLETE: {total_scored_rows:,} pixels scored in {(time.time() - t_infer_start)/60:.2f} mins.")
    print("==============================================================================\n")

    # 4. Global Consolidation & Aggregation across All Chunks
    print(f"[GLOBAL AGGREGATION] Rolling up {len(chunk_accumulators):,} chunk summaries...")
    all_chunks_df = pd.concat(chunk_accumulators, ignore_index=True)

    # Combine chunk aggregations into final segment statistics
    final_agg = all_chunks_df.groupby(["road_id", "year"]).agg(
        total_sum_p=("sum_prob", "sum"),
        total_sum_sq_p=("sum_sq_prob", "sum"),
        total_pixels=("n_pixels", "sum"),
        median_prob=("median_prob", "mean")  # Average chunk medians
    ).reset_index()

    final_agg["mean_prob"] = final_agg["total_sum_p"] / final_agg["total_pixels"]

    # Compute population standard deviation: sqrt(E[X^2] - (E[X])^2)
    var_prob = (final_agg["total_sum_sq_p"] / final_agg["total_pixels"]) - (final_agg["mean_prob"] ** 2)
    final_agg["std_prob"] = np.sqrt(np.maximum(0.0, var_prob))
    final_agg["n_pixels"] = final_agg["total_pixels"].astype(int)

    final_table = final_agg[["road_id", "year", "mean_prob", "median_prob", "std_prob", "n_pixels"]].copy()

    # 5. Merge Ground Truth Labels Metadata (if available)
    if labels_meta is not None:
        print("[METADATA] Joining ground-truth road inspection labels...")
        final_table = final_table.merge(labels_meta, on="road_id", how="left")

    final_table = final_table.sort_values(["road_id", "year"]).reset_index(drop=True)

    # 6. Save Final Outputs
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(out_dir, exist_ok=True)

    final_table.to_csv(args.output_csv, index=False)
    file_size_kb = os.path.getsize(args.output_csv) / 1024
    print(f"\n[ARTIFACT] Segment predictions saved to: {args.output_csv}")
    print(f"[ARTIFACT] Summary table contains: {len(final_table):,} road-year observations across {final_table['road_id'].nunique():,} unique roads.")
    print(f"[ARTIFACT] File size: {file_size_kb:.1f} KB")

    if args.output_parquet:
        final_table.to_parquet(args.output_parquet, index=False)
        print(f"[ARTIFACT] Parquet version saved to: {args.output_parquet}")

    elapsed_total = (time.time() - start_total) / 60
    print(f"\n=== [FINISHED] Segment-level aggregation completed in {elapsed_total:.2f} mins. ===\n")


if __name__ == "__main__":
    main()
