"""
src/data/loader.py
PyArrow and memory-mapped ingestion engine implementing the Master Schema Contract
and ground-truth road verification logic for national-scale training on FUCHS-CSC.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config.config import METADATA_COLUMNS, MODEL_FEATURE_REGISTRY

# Pure control road surface classifications
PURE_CONTROL_SURFACES: Set[str] = {
    "Earth Graded", "Earth Sand", "Earth Track", "Proclaimed Only"
}

def load_ground_truth_labels(shp_path: Path) -> pd.DataFrame:
    """
    Parses DistrictRoads shapefile, normalizes audit fields, and derives clean verification windows.
    """
    if not shp_path.exists():
        raise FileNotFoundError(f"Roads shapefile not found at: {shp_path}")

    gdf = gpd.read_file(shp_path)
    labels = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

    # Standardize column naming
    if "segment_id" in labels.columns and "road_id" not in labels.columns:
        labels = labels.rename(columns={"segment_id": "road_id"})
    labels["road_id"] = labels["road_id"].astype(str)

    # Clean human audit columns
    labels = labels.rename(columns={
        "~man_start": "man_start_approx",
        "`~man_start`": "man_start_approx",
        "~man_end": "man_end_approx",
        "`~man_end`": "man_end_approx"
    })

    # Filter to verified inspection codes (1: Confirmed, 2/6: Approx, 4/5: Control, 9: Always Paved)
    labels = labels[labels["man_check"].isin([1, 2, 4, 5, 6, 9])].copy()

    def parse_year(val) -> float:
        try:
            return float(val) if pd.notna(val) else np.nan
        except (ValueError, TypeError):
            return np.nan

    for col in ["true_start", "true_end", "man_start_approx", "man_end_approx"]:
        if col in labels.columns:
            labels[f"{col}_year"] = labels[col].apply(parse_year)

    # Assign effective construction window boundaries
    labels["start_year"] = np.where(
        labels["man_check"] == 1,
        labels.get("true_start_year", np.nan),
        labels.get("man_start_approx_year", np.nan)
    )
    labels["end_year"] = np.where(
        labels["man_check"] == 1,
        labels.get("true_end_year", np.nan),
        labels.get("man_end_approx_year", np.nan)
    )

    labels["is_pure_control"] = labels["SURFACE_TY"].isin(PURE_CONTROL_SURFACES)
    keep_cols = ["road_id", "man_check", "SURFACE_TY", "start_year", "end_year", "is_pure_control"]
    return labels[[c for c in keep_cols if c in labels.columns]].copy()


def apply_ground_truth_labeling(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    cutoff_year_control: int = 2020
) -> pd.DataFrame:
    """
    Applies the 3-tier ground-truth labeling rule:
    1. Treatment Roads (man_check in {1,2,6}): drop active construction window,
       label Pre_Construction if year < start, Post_Construction if year > end.
    2. Pure Control Roads: label Pre_Construction up to cutoff_year_control (2020).
    3. Always Paved Roads (man_check == 9): label Post_Construction for year >= 1990.
    """
    merged = df.merge(labels_df, on="road_id", how="inner")
    partitions: List[pd.DataFrame] = []

    # 1. Construction Roads: Exclude transition noise during construction window
    treat_mask = (
        merged["man_check"].isin([1, 2, 6]) &
        merged["start_year"].notna() &
        merged["end_year"].notna()
    )
    treat_df = merged[treat_mask].copy()
    # Exclude observations strictly inside active construction window [start_year, end_year]
    clean_treat = treat_df[(treat_df["year"] < treat_df["start_year"]) | (treat_df["year"] > treat_df["end_year"])].copy()
    clean_treat["label"] = np.where(
        clean_treat["year"] > clean_treat["end_year"],
        "Post_Construction",
        "Pre_Construction"
    )
    partitions.append(clean_treat)

    # 2. Pure Controls: Earth/gravel roads never upgraded
    ctrl_mask = (
        (merged["man_check"].isin([4, 5]) | merged["is_pure_control"]) &
        (merged["year"] <= cutoff_year_control)
    )
    ctrl_df = merged[ctrl_mask].copy()
    ctrl_df["label"] = "Pre_Construction"
    partitions.append(ctrl_df)

    # 3. Always Bitumen/Paved (Historical controls)
    always_mask = (merged["man_check"] == 9) & (merged["year"] >= 1990)
    always_df = merged[always_mask].copy()
    always_df["label"] = "Post_Construction"
    partitions.append(always_df)

    if not partitions:
        return pd.DataFrame()

    labeled_data = pd.concat(partitions, ignore_index=True)
    return labeled_data.dropna(subset=["label"])


def load_harmonized_national_dataset(
    features_dir: Path,
    feature_cols: List[str],
    labeled_road_ids: Set[str],
    batch_size_files: int = 5
) -> pd.DataFrame:
    """
    Iterates over all 35 regional Parquet files, enforces the Master Schema Contract,
    imputes missing zero-variance buffer columns with float32 0.0, downcasts features to float32,
    and filters rows to labeled road IDs to conserve RAM on FUCHS-CSC.
    """
    parquet_files = sorted(glob.glob(str(features_dir / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet feature files found in {features_dir}")

    required_cols_set = set(METADATA_COLUMNS + feature_cols)
    dfs: List[pd.DataFrame] = []

    print(f"[DATA LOADER] Ingesting {len(parquet_files)} regional Parquet files from: {features_dir}")
    print(f"[DATA LOADER] Master Schema Contract: Enforcing {len(feature_cols)} active features.")

    for idx, filepath in enumerate(parquet_files, 1):
        schema = pq.read_schema(filepath)
        existing_cols = set(schema.names)

        # Select only needed columns present in this file
        cols_to_read = [c for c in (METADATA_COLUMNS + feature_cols) if c in existing_cols]

        # Read PyArrow table and immediately convert to DataFrame
        table = pq.read_table(filepath, columns=cols_to_read)
        df_chunk = table.to_pandas()
        df_chunk["road_id"] = df_chunk["road_id"].astype(str)

        # Filter to labeled roads early to minimize memory footprint
        df_chunk = df_chunk[df_chunk["road_id"].isin(labeled_road_ids)].copy()

        if len(df_chunk) == 0:
            continue

        # MASTER SCHEMA HARMONIZATION: Impute missing buffer columns with float32 zeros
        missing_features = [c for c in feature_cols if c not in df_chunk.columns]
        if missing_features:
            for missing_col in missing_features:
                df_chunk[missing_col] = np.float32(0.0)

        # Downcast all feature columns strictly to float32
        for col in feature_cols:
            df_chunk[col] = df_chunk[col].astype(np.float32)

        # Optimize metadata dtypes
        df_chunk["year"] = df_chunk["year"].astype(np.int16)
        df_chunk["half"] = df_chunk["half"].astype(np.int8)

        dfs.append(df_chunk)
        if idx % batch_size_files == 0 or idx == len(parquet_files):
            print(f"[DATA LOADER] Processed {idx}/{len(parquet_files)} files | Current in-memory rows: {sum(len(d) for d in dfs):,}")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[DATA LOADER] Ingestion complete. Total raw candidate rows: {len(full_df):,}")
    return full_df


def prepare_training_matrices(
    full_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Merges data with labels, handles temporal NaN drops, and extracts C-contiguous float32 arrays.
    Returns:
        X (np.ndarray): Shape (N, D), float32 feature matrix
        y (np.ndarray): Shape (N,), int8 binary target array (0=Pre, 1=Post)
        road_ids (np.ndarray): Shape (N,), string array for GroupKFold grouping
        labeled_df (pd.DataFrame): Full DataFrame with metadata for evaluation tracking
    """
    labeled_df = apply_ground_truth_labeling(full_df, labels_df)

    # For temporal models (M6/M7), drop rows with NaN in lag columns (e.g., year 1990)
    nan_mask = labeled_df[feature_cols].isna().any(axis=1)
    if nan_mask.sum() > 0:
        print(f"[DATA LOADER] Dropping {nan_mask.sum():,} rows containing NaN in temporal features (e.g. 1990 boundary).")
        labeled_df = labeled_df[~nan_mask].copy()

    print(f"[DATA LOADER] Final labeled training rows: {len(labeled_df):,} across {labeled_df['road_id'].nunique()} roads.")

    X = np.ascontiguousarray(labeled_df[feature_cols].values, dtype=np.float32)
    y = np.where(labeled_df["label"] == "Post_Construction", 1, 0).astype(np.int8)
    road_ids = labeled_df["road_id"].values

    return X, y, road_ids, labeled_df
