#!/bin/bash
# ==============================================================================
# SLURM BATCH SCRIPT: PARALLEL JOB ARRAY CROSS-VALIDATION (MODEL M5)
# CLUSTER: FUCHS-CSC (Goethe University Frankfurt)
# ==============================================================================
#SBATCH --job-name=rf_m5_arr
#SBATCH --partition=fuchs
#SBATCH --array=0-4                 # 5 Array Tasks: Folds 0, 1, 2, 3, 4 in parallel
#SBATCH --nodes=1                   # 1 node per array task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20          # 20 cores per fold
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=03:00:00             # Each individual fold completes in ~1.5 hours
#SBATCH --output=/work/fuchs/agmisc/cmiel/road_ml/logs/m5_arr_%A_%a.out
#SBATCH --error=/work/fuchs/agmisc/cmiel/road_ml/logs/m5_arr_%A_%a.err

set -euo pipefail

FOLD_ID=${SLURM_ARRAY_TASK_ID}
echo "=== [ARRAY TASK START] Job ID: ${SLURM_ARRAY_JOB_ID} | Task/Fold: ${FOLD_ID} ==="
echo "Node: $(hostname) | Cores: ${SLURM_CPUS_PER_TASK}"

# Thread Concurrency Control
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT_DIR="/work/fuchs/agmisc/cmiel/road_ml"
CONTAINER_SIF="${PROJECT_DIR}/containers/road_ml.sif"
DATA_TAR="${PROJECT_DIR}/data/national_features_v1.tar"
ROADS_SHP="${PROJECT_DIR}/data/shapefiles/DistrictRoads_with training data.shp"
OUTPUT_DIR="${PROJECT_DIR}/results/m5_array_${SLURM_ARRAY_JOB_ID}"

LOCAL_SCRATCH="/localscratch/${SLURM_JOB_ID}"
if [ ! -d "/localscratch" ]; then
    LOCAL_SCRATCH="/tmp/${SLURM_JOB_ID}"
fi
export TMPDIR="${LOCAL_SCRATCH}"
export JOBLIB_TEMP_FOLDER="${LOCAL_SCRATCH}/joblib_cache"

mkdir -p "${LOCAL_SCRATCH}/data"
mkdir -p "${LOCAL_SCRATCH}/output"
mkdir -p "${JOBLIB_TEMP_FOLDER}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

cleanup_scratch() {
    rm -rf "${LOCAL_SCRATCH}" || true
}
trap cleanup_scratch EXIT TERM INT HUP USR1

# Stage-In
if [ -f "${DATA_TAR}" ]; then
    echo "[STAGE-IN] Extracting national dataset from ${DATA_TAR}..."
    tar -C "${LOCAL_SCRATCH}/data" -xf "${DATA_TAR}"
elif [ -d "${PROJECT_DIR}/data/parquet_national" ]; then
    echo "[STAGE-IN] Staging Parquet files..."
    cp -r "${PROJECT_DIR}/data/parquet_national"/* "${LOCAL_SCRATCH}/data/"
fi

DATA_INPUT="${LOCAL_SCRATCH}/data"
if [ ! "$(ls -A "${DATA_INPUT}" 2>/dev/null)" ]; then
    DATA_INPUT="${PROJECT_DIR}/data/parquet_national"
fi

# Execute Isolated Fold
echo "[EXECUTION] Training Fold ${FOLD_ID} on 20 Cores..."
if [ -f "${CONTAINER_SIF}" ]; then
    apptainer exec \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --bind "${LOCAL_SCRATCH}:${LOCAL_SCRATCH}" \
        "${CONTAINER_SIF}" \
        numactl --interleave=all python "${PROJECT_DIR}/train_rf.py" \
            --model m5 \
            --mode single_fold \
            --fold ${FOLD_ID} \
            --scope national \
            --features-dir "${DATA_INPUT}" \
            --roads-shp "${ROADS_SHP}" \
            --output-dir "${LOCAL_SCRATCH}/output" \
            --scratch-dir "${LOCAL_SCRATCH}" \
            --n-trees 500 \
            --max-depth 25 \
            --n-jobs ${SLURM_CPUS_PER_TASK} \
            --backend threading
else
    if [ -f "${HOME}/my_environments/geo_env/bin/activate" ]; then
        source "${HOME}/my_environments/geo_env/bin/activate"
    elif [ -f "${PROJECT_DIR}/venv/bin/activate" ]; then
        source "${PROJECT_DIR}/venv/bin/activate"
    fi
    numactl --interleave=all python "${PROJECT_DIR}/train_rf.py" \
        --model m5 \
        --mode single_fold \
        --fold ${FOLD_ID} \
        --scope national \
        --features-dir "${DATA_INPUT}" \
        --roads-shp "${ROADS_SHP}" \
        --output-dir "${LOCAL_SCRATCH}/output" \
        --scratch-dir "${LOCAL_SCRATCH}" \
        --n-trees 500 \
        --max-depth 25 \
        --n-jobs ${SLURM_CPUS_PER_TASK} \
        --backend threading
fi

# Stage-Out Fold Artifacts
echo "[STAGE-OUT] Syncing Fold ${FOLD_ID} outputs..."
rsync -av "${LOCAL_SCRATCH}/output/" "${OUTPUT_DIR}/"

echo "=== [ARRAY TASK COMPLETE] Fold ${FOLD_ID} Finished ==="
