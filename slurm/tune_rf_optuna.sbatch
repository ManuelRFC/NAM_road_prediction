#!/bin/bash
# ==============================================================================
# SLURM BATCH SCRIPT: BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA)
# CLUSTER: FUCHS-CSC (Goethe University Frankfurt)
# ==============================================================================
#SBATCH --job-name=rf_tune_optuna
#SBATCH --partition=fuchs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/project/road_ml/logs/tune_%j.out
#SBATCH --error=/beegfs/project/road_ml/logs/tune_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_DIR="/beegfs/project/road_ml"
CONTAINER_SIF="${PROJECT_DIR}/containers/road_ml.sif"
DATA_TAR="${PROJECT_DIR}/data/national_features_v1.tar"
ROADS_SHP="${PROJECT_DIR}/data/shapefiles/DistrictRoads_with training data.shp"
OUTPUT_DIR="${PROJECT_DIR}/results/tuning_${SLURM_JOB_ID}"

LOCAL_SCRATCH="/localscratch/${SLURM_JOB_ID}"
export TMPDIR="${LOCAL_SCRATCH}"
export JOBLIB_TEMP_FOLDER="${LOCAL_SCRATCH}/joblib_cache"

mkdir -p "${LOCAL_SCRATCH}/data"
mkdir -p "${JOBLIB_TEMP_FOLDER}"
mkdir -p "${OUTPUT_DIR}"

cleanup_scratch() {
    rm -rf "${LOCAL_SCRATCH}" || true
}
trap cleanup_scratch EXIT TERM INT HUP USR1

echo "[STAGE-IN] Extracting features..."
tar -C "${LOCAL_SCRATCH}/data" -xf "${DATA_TAR}"

echo "[TUNING] Launching 50-Trial Optuna TPE Search on Model M5..."
apptainer exec \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${LOCAL_SCRATCH}:${LOCAL_SCRATCH}" \
    "${CONTAINER_SIF}" \
    numactl --interleave=all python "${PROJECT_DIR}/tune_rf.py" \
        --model m5 \
        --trials 50 \
        --features-dir "${LOCAL_SCRATCH}/data" \
        --roads-shp "${ROADS_SHP}" \
        --output-dir "${OUTPUT_DIR}" \
        --n-jobs ${SLURM_CPUS_PER_TASK}

echo "=== [TUNING COMPLETE] Results saved in ${OUTPUT_DIR} ==="
