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
#SBATCH --output=/work/fuchs/agmisc/cmiel/road_ml/logs/tune_%j.out
#SBATCH --error=/work/fuchs/agmisc/cmiel/road_ml/logs/tune_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CODE_DIR="${HOME}/road_ml"
WORK_DIR="/work/fuchs/agmisc/cmiel/road_ml"
CONTAINER_SIF="${WORK_DIR}/containers/road_ml.sif"
DATA_TAR="${WORK_DIR}/data/national_features_v1.tar"
ROADS_SHP="${WORK_DIR}/data/shapefiles/DistrictRoads_with training data.shp"
OUTPUT_DIR="${WORK_DIR}/results/tuning_${SLURM_JOB_ID}"

LOCAL_SCRATCH="/localscratch/${SLURM_JOB_ID}"
if [ ! -d "/localscratch" ]; then
    LOCAL_SCRATCH="/tmp/${SLURM_JOB_ID}"
fi
export TMPDIR="${LOCAL_SCRATCH}"
export JOBLIB_TEMP_FOLDER="${LOCAL_SCRATCH}/joblib_cache"

mkdir -p "${LOCAL_SCRATCH}/data"
mkdir -p "${JOBLIB_TEMP_FOLDER}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORK_DIR}/logs"

cleanup_scratch() {
    rm -rf "${LOCAL_SCRATCH}" || true
}
trap cleanup_scratch EXIT TERM INT HUP USR1

if [ -f "${DATA_TAR}" ]; then
    echo "[STAGE-IN] Extracting features..."
    tar -C "${LOCAL_SCRATCH}/data" -xf "${DATA_TAR}"
elif [ -d "${WORK_DIR}/data/parquet_national" ]; then
    echo "[STAGE-IN] Copying Parquet files..."
    cp -r "${WORK_DIR}/data/parquet_national"/* "${LOCAL_SCRATCH}/data/"
fi

DATA_INPUT="${LOCAL_SCRATCH}/data"
if [ ! "$(ls -A "${DATA_INPUT}" 2>/dev/null)" ]; then
    DATA_INPUT="${WORK_DIR}/data/parquet_national"
fi

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

echo "[TUNING] Launching 50-Trial Optuna TPE Search on Model M5..."
if [ -f "${CONTAINER_SIF}" ]; then
    apptainer exec \
        --bind "${WORK_DIR}:${WORK_DIR}" \
        --bind "${CODE_DIR}:${CODE_DIR}" \
        --bind "${LOCAL_SCRATCH}:${LOCAL_SCRATCH}" \
        "${CONTAINER_SIF}" \
        numactl --interleave=all python "${CODE_DIR}/tune_rf.py" \
            --model m5 \
            --trials 50 \
            --features-dir "${DATA_INPUT}" \
            --roads-shp "${ROADS_SHP}" \
            --output-dir "${OUTPUT_DIR}" \
            --n-jobs ${SLURM_CPUS_PER_TASK}
else
    if [ -f "${HOME}/my_environments/geo_env/bin/activate" ]; then
        source "${HOME}/my_environments/geo_env/bin/activate"
    elif [ -f "${WORK_DIR}/venv/bin/activate" ]; then
        source "${WORK_DIR}/venv/bin/activate"
    fi
    numactl --interleave=all python "${CODE_DIR}/tune_rf.py" \
        --model m5 \
        --trials 50 \
        --features-dir "${DATA_INPUT}" \
        --roads-shp "${ROADS_SHP}" \
        --output-dir "${OUTPUT_DIR}" \
        --n-jobs ${SLURM_CPUS_PER_TASK}
fi

echo "=== [TUNING COMPLETE] Results saved in ${OUTPUT_DIR} ==="
