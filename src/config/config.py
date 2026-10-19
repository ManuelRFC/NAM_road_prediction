"""
src/config/config.py
Authoritative configuration schemas, feature taxonomies, and hardware contracts for FUCHS-CSC.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

# ==============================================================================
# FEATURE CATALOG TAXONOMY (MODELS M1 THROUGH M7)
# ==============================================================================

RAW_SPECTRAL_BANDS: List[str] = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"
]

SPECTRAL_INDICES: List[str] = [
    "NDVI", "NDBI", "TC_Brightness", "TC_Greenness", "TC_Wetness"
]

TOPOGRAPHY_FEATURES: List[str] = [
    "Elevation_m", "Slope_deg"
]

# 11 Core physical variables used for buffer contrasts
BUFFER_BASE_VARIABLES: List[str] = RAW_SPECTRAL_BANDS + SPECTRAL_INDICES
BUFFER_TIERS: List[str] = ["Buf_0_45", "Buf_45_79", "Buf_79_Plus"]

# 33 Delta Contrast Features (v_pixel - Mean_buffer)
BUFFER_DELTA_FEATURES: List[str] = [
    f"Delta_{var}_{tier}" for var in BUFFER_BASE_VARIABLES for tier in BUFFER_TIERS
]

# 33 Z-Score Standardized Contrast Features ((v_pixel - Mean_buffer) / SD_buffer)
BUFFER_Z_FEATURES: List[str] = [
    f"Z_{var}_{tier}" for var in BUFFER_BASE_VARIABLES for tier in BUFFER_TIERS
]

# M4 Baseline: 6 Raw + 5 Indices + 2 Topo + 33 Deltas + 33 Z-scores = 79 features
M4_FEATURES: List[str] = (
    RAW_SPECTRAL_BANDS + SPECTRAL_INDICES + TOPOGRAPHY_FEATURES +
    BUFFER_DELTA_FEATURES + BUFFER_Z_FEATURES
)

# M5: M4 (79) + 11 Road-Segment-Level Spatial Means = 90 features
SEGMENT_MEAN_FEATURES: List[str] = [
    f"Seg_Mean_{var}" for var in BUFFER_BASE_VARIABLES
]
M5_FEATURES: List[str] = M4_FEATURES + SEGMENT_MEAN_FEATURES

# M6: M4 (79) + 11 Lag-1 differences + 11 YoY seasonal differences = 101 features
TEMPORAL_LAG1_FEATURES: List[str] = [f"Lag1_{var}" for var in BUFFER_BASE_VARIABLES]
TEMPORAL_YOY_FEATURES: List[str] = [f"YoY_{var}" for var in BUFFER_BASE_VARIABLES]
M6_FEATURES: List[str] = M4_FEATURES + TEMPORAL_LAG1_FEATURES + TEMPORAL_YOY_FEATURES

# M7: Full Comprehensive Model (M4 + 11 Segment + 22 Temporal) = 112 features
M7_FEATURES: List[str] = (
    M4_FEATURES + SEGMENT_MEAN_FEATURES + TEMPORAL_LAG1_FEATURES + TEMPORAL_YOY_FEATURES
)

METADATA_COLUMNS: List[str] = [
    "road_id", "year", "half", "month_start", "month_end",
    "lon", "lat", "length_m_new", "dist_m_new"
]

MODEL_FEATURE_REGISTRY: Dict[str, List[str]] = {
    "m1": RAW_SPECTRAL_BANDS,
    "m2": RAW_SPECTRAL_BANDS + SPECTRAL_INDICES,
    "m3": RAW_SPECTRAL_BANDS + SPECTRAL_INDICES + TOPOGRAPHY_FEATURES,
    "m4": M4_FEATURES,
    "m5": M5_FEATURES,
    "m6": M6_FEATURES,
    "m7": M7_FEATURES,
}

# ==============================================================================
# DATACLASS CONFIGURATIONS
# ==============================================================================

@dataclass
class PathConfig:
    """Filesystem path configuration across FUCHS-CSC storage tiers."""
    beegfs_root: Path = Path("/beegfs/project/road_ml")
    features_dir: Path = Path("/beegfs/project/road_ml/data/parquet_national")
    roads_shp: Path = Path("/beegfs/project/road_ml/data/shapefiles/DistrictRoads_with training data.shp")
    scratch_dir: Path = Path(os.environ.get("TMPDIR", "/localscratch"))
    output_dir: Path = Path("/beegfs/project/road_ml/results")
    archive_tar: Path = Path("/beegfs/project/road_ml/data/national_features_v1.tar")

@dataclass
class RFHyperparameters:
    """Random Forest hyperparameter configuration with FUCHS-CSC defaults."""
    n_estimators: int = 500
    max_depth: Optional[int] = 25
    min_samples_split: int = 10
    min_samples_leaf: int = 4
    max_features: str = "sqrt"
    criterion: str = "entropy"
    class_weight: str = "balanced"
    bootstrap: bool = True
    random_state: int = 42

@dataclass
class HardwareConfig:
    """FUCHS-CSC compute node allocation parameters."""
    n_jobs: int = int(os.environ.get("SLURM_CPUS_PER_TASK", 20))
    joblib_backend: str = "threading"  # 'threading' or 'loky'
    memory_limit_gb: float = 120.0
    numa_interleave: bool = True
    joblib_temp_folder: Path = Path(os.environ.get("TMPDIR", "/localscratch")) / "joblib_cache"

@dataclass
class PipelineConfig:
    """Master pipeline execution configuration."""
    model_name: str = "m5"
    scope: str = "national"  # 'national' or 'o_regions'
    mode: str = "cv"         # 'cv', 'full', 'single_fold'
    fold: Optional[int] = None
    k_folds: int = 5
    test_fraction: float = 0.15
    paths: PathConfig = field(default_factory=PathConfig)
    rf_params: RFHyperparameters = field(default_factory=RFHyperparameters)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def get_features(self) -> List[str]:
        model_key = self.model_name.lower()
        if model_key not in MODEL_FEATURE_REGISTRY:
            raise ValueError(f"Unknown model: {self.model_name}. Choose from {list(MODEL_FEATURE_REGISTRY.keys())}")
        return MODEL_FEATURE_REGISTRY[model_key]
