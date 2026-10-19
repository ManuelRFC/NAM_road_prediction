import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Mock Heavy C++ Dependencies (geopandas, shapely, pyarrow) for Windows ARM64
from unittest.mock import MagicMock
sys.modules['geopandas'] = MagicMock()
sys.modules['shapely'] = MagicMock()
sys.modules['shapely.geometry'] = MagicMock()

mock_pa = MagicMock()
class FakeTable: pass
class FakeChunkedArray: pass
class FakeArray: pass
class FakeRecordBatch: pass
mock_pa.Table = FakeTable
mock_pa.ChunkedArray = FakeChunkedArray
mock_pa.Array = FakeArray
mock_pa.RecordBatch = FakeRecordBatch
sys.modules['pyarrow'] = mock_pa
sys.modules['pyarrow.parquet'] = MagicMock()

# Import our custom modules
from src.config.config import PipelineConfig, PathConfig, HardwareConfig, M5_FEATURES, METADATA_COLUMNS
import src.data.loader as loader
from src.data.loader import load_harmonized_national_dataset, prepare_training_matrices, apply_ground_truth_labeling
from src.models.trainer import train_single_fold
import pyarrow.parquet as pq

# Mock pyarrow functionality
def mock_read_schema(filepath):
    mock_schema = MagicMock()
    mock_schema.names = METADATA_COLUMNS + M5_FEATURES
    return mock_schema

def mock_read_table(filepath, columns):
    mock_table = MagicMock()
    # Read the CSV we saved as dummy data
    df = pd.read_csv(filepath)
    # Ensure columns match expected
    missing = [c for c in columns if c not in df.columns]
    for m in missing:
        df[m] = 0.0
    mock_table.to_pandas.return_value = df[columns]
    return mock_table

pq.read_schema = mock_read_schema
pq.read_table = mock_read_table


def create_dummy_data(temp_dir: Path):
    """Creates a tiny dummy dataset to mimic the cluster environment."""
    print("Generating dummy data (using CSV under the hood to bypass PyArrow compiler errors)...")
    features_dir = temp_dir / "parquet_national"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy features
    rows = []
    for road in ["R1", "R2", "R3", "R4", "R5"]:
        for year in range(1990, 2000):
            row = {
                "road_id": road,
                "year": year,
                "half": 1,
                "month_start": 1,
                "month_end": 6,
                "lon": 0.0,
                "lat": 0.0,
                "length_m_new": 100.0,
                "dist_m_new": 0.0
            }
            for feat in M5_FEATURES:
                row[feat] = np.random.rand()
            rows.append(row)
            
    df_features = pd.DataFrame(rows)
    # Save as CSV instead of parquet, but keep the .parquet extension for the glob
    df_features.to_csv(features_dir / "dummy_region_1.parquet", index=False)
    
    return features_dir


def run_smoke_test():
    print("=== STARTING LOCAL SMOKE TEST ===")
    
    temp_dir = Path("./local_test_data")
    temp_dir.mkdir(exist_ok=True)
    
    features_dir = create_dummy_data(temp_dir)
    dummy_shp_path = temp_dir / "dummy.shp"
    
    local_paths = PathConfig(
        beegfs_root=temp_dir,
        features_dir=features_dir,
        roads_shp=dummy_shp_path,
        scratch_dir=temp_dir,
        output_dir=temp_dir / "results"
    )
    
    local_hardware = HardwareConfig(
        n_jobs=2,
        joblib_backend="threading",
        memory_limit_gb=2.0,
        numa_interleave=False,
        joblib_temp_folder=temp_dir
    )
    
    config = PipelineConfig(
        model_name="m5",
        scope="national",
        mode="single_fold",
        paths=local_paths,
        hardware=local_hardware,
        k_folds=2
    )
    
    print("Mocking Ground Truth Labels...")
    labels_df = pd.DataFrame({
        "road_id": ["R1", "R2", "R3", "R4", "R5"],
        "man_check": [1, 1, 4, 9, 2],
        "SURFACE_TY": ["Earth Graded", "Earth Graded", "Earth Sand", "Bitumen", "Earth Graded"],
        "start_year": [2010.0, 2015.0, np.nan, np.nan, 2018.0],
        "end_year": [2011.0, 2016.0, np.nan, np.nan, 2019.0],
        "is_pure_control": [False, False, True, False, False]
    })
    loader.load_ground_truth_labels = lambda x: labels_df
    
    full_df = load_harmonized_national_dataset(
        features_dir=features_dir,
        feature_cols=config.get_features(),
        labeled_road_ids=set(labels_df["road_id"].unique()),
        batch_size_files=1
    )
    
    X, y, road_ids, labeled_df = prepare_training_matrices(
        full_df=full_df,
        labels_df=labels_df,
        feature_cols=config.get_features()
    )
    print(f"Matrix shape: X={X.shape}, y={y.shape}")
    
    print("Testing Random Forest training loop...")
    results = train_single_fold(
        X=X,
        y=y,
        road_ids=road_ids,
        target_fold=0,
        feature_names=config.get_features(),
        config=config
    )
    
    print("\n=== SMOKE TEST PASSED ===")
    print(f"Model trained successfully! Fold 0 Accuracy: {results['metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    run_smoke_test()
