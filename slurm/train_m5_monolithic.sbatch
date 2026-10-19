#!/bin/bash
# ==============================================================================
# SLURM BATCH SCRIPT: MONOLITHIC MODEL M5 TRAINING (5-FOLD CV + PRODUCTION FIT)
# CLUSTER: FUCHS-CSC (Goethe University Frankfurt)
# ==============================================================================
#SBATCH --job-name=rf_m5_mono
#SBATCH --partition=fuchs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/work/fuchs/agmisc/cmiel/road_ml/logs/m5_mono_%j.out
#SBATCH --error=/work/fuchs/agmisc/cmiel/road_ml/logs/m5_mono_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cmiel@em.uni-frankfurt.de

set -euo pipefail

echo "=== [JOB START] FUCHS-CSC SLURM ID: ${SLURM_JOB_ID} ==="
echo "Node: $(hostname) | Allocated Cores: ${SLURM_CPUS_PER_TASK}"
echo "Start Time: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# 1. Thread Management: Prevent 400-thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# 2. Filesystem & Storage Hierarchy Setup
PROJECT_DIR="/work/fuchs/agmisc/cmiel/road_ml"
CONTAINER_SIF="${PROJECT_DIR}/containers/road_ml.sif"
DATA_TAR="${PROJECT_DIR}/data/national_features_v1.tar"
ROADS_SHP="${PROJECT_DIR}/data/shapefiles/DistrictRoads_with training data.shp"
OUTPUT_DIR="${PROJECT_DIR}/results/m5_monolithic_${SLURM_JOB_ID}"

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

# 3. Robust POSIX Signal Trap for Zero-Orphan Scratch Cleanup
cleanup_scratch() {
    local exit_code=$?
    echo "[TRAP] Job exiting with code ${exit_code}. Purging local scratch: ${LOCAL_SCRATCH}"
    if [ -d "${LOCAL_SCRATCH}/output" ] && [ "$(ls -A "${LOCAL_SCRATCH}/output" 2>/dev/null)" ]; then
        echo "[TRAP] Syncing remaining artifacts to work storage..."
        rsync -av "${LOCAL_SCRATCH}/output/" "${OUTPUT_DIR}/" || true
    fi
    rm -rf "${LOCAL_SCRATCH}" || true
    echo "[TRAP] Cleanup complete."
    exit ${exit_code}
}
trap cleanup_scratch EXIT TERM INT HUP USR1

# 4. Sequential High-Speed Stage-In Protocol
if [ -f "${DATA_TAR}" ]; then
    echo "[STAGE-IN] Extracting uncompressed national archive from ${DATA_TAR} to scratch..."
    t_stage_start=$(date +%s)
    tar -C "${LOCAL_SCRATCH}/data" -xf "${DATA_TAR}"
    t_stage_elapsed=$(( $(date +%s) - t_stage_start ))
    echo "[STAGE-IN] Completed transfer in ${t_stage_elapsed} seconds."
elif [ -d "${PROJECT_DIR}/data/parquet_national" ]; then
    echo "[STAGE-IN] Staging Parquet files from ${PROJECT_DIR}/data/parquet_national to local scratch..."
    cp -r "${PROJECT_DIR}/data/parquet_national"/* "${LOCAL_SCRATCH}/data/"
else
    echo "[STAGE-IN] Using ${PROJECT_DIR}/data directly."
    LOCAL_SCRATCH_DATA="${PROJECT_DIR}/data"
fi

DATA_INPUT="${LOCAL_SCRATCH}/data"
if [ ! "$(ls -A "${DATA_INPUT}" 2>/dev/null)" ]; then
    DATA_INPUT="${PROJECT_DIR}/data/parquet_national"
fi

# 5. Execute Training (Apptainer OR Native Python Environment)
echo "[EXECUTION] Launching Model M5 Training with NUMA Interleaving..."
if [ -f "${CONTAINER_SIF}" ]; then
    apptainer exec \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --bind "${LOCAL_SCRATCH}:${LOCAL_SCRATCH}" \
        "${CONTAINER_SIF}" \
        numactl --interleave=all python "${PROJECT_DIR}/train_rf.py" \
            --model m5 \
            --mode cv \
            --scope national \
            --features-dir "${DATA_INPUT}" \
            --roads-shp "${ROADS_SHP}" \
            --output-dir "${LOCAL_SCRATCH}/output" \
            --scratch-dir "${LOCAL_SCRATCH}" \
            --n-trees 500 \
            --max-depth 25 \
            --min-samples-split 10 \
            --min-samples-leaf 4 \
            --criterion entropy \
            --class-weight balanced \
            --n-jobs ${SLURM_CPUS_PER_TASK} \
            --backend threading
else
    if [ -f "${HOME}/my_environments/geo_env/bin/activate" ]; then
        echo "[ENV] Activating existing geo_env..."
        source "${HOME}/my_environments/geo_env/bin/activate"
    elif [ -f "${PROJECT_DIR}/venv/bin/activate" ]; then
        echo "[ENV] Activating project venv..."
        source "${PROJECT_DIR}/venv/bin/activate"
    fi
    numactl --interleave=all python "${PROJECT_DIR}/train_rf.py" \
        --model m5 \
        --mode cv \
        --scope national \
        --features-dir "${DATA_INPUT}" \
        --roads-shp "${ROADS_SHP}" \
        --output-dir "${LOCAL_SCRATCH}/output" \
        --scratch-dir "${LOCAL_SCRATCH}" \
        --n-trees 500 \
        --max-depth 25 \
        --min-samples-split 10 \
        --min-samples-leaf 4 \
        --criterion entropy \
        --class-weight balanced \
        --n-jobs ${SLURM_CPUS_PER_TASK} \
        --backend threading
fi

# 6. Stage-Out Protocol: Copy Artifacts to Storage
echo "[STAGE-OUT] Syncing model checkpoints and metrics..."
rsync -av "${LOCAL_SCRATCH}/output/" "${OUTPUT_DIR}/"

echo "=== [JOB COMPLETE] Finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
