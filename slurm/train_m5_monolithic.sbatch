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
#SBATCH --output=/beegfs/project/road_ml/logs/m5_mono_%j.out
#SBATCH --error=/beegfs/project/road_ml/logs/m5_mono_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@em.uni-frankfurt.de

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
PROJECT_DIR="/beegfs/project/road_ml"
CONTAINER_SIF="${PROJECT_DIR}/containers/road_ml.sif"
DATA_TAR="${PROJECT_DIR}/data/national_features_v1.tar"
ROADS_SHP="${PROJECT_DIR}/data/shapefiles/DistrictRoads_with training data.shp"
OUTPUT_DIR="${PROJECT_DIR}/results/m5_monolithic_${SLURM_JOB_ID}"

LOCAL_SCRATCH="/localscratch/${SLURM_JOB_ID}"
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
    # Salvage any outputs before deletion
    if [ -d "${LOCAL_SCRATCH}/output" ] && [ "$(ls -A "${LOCAL_SCRATCH}/output" 2>/dev/null)" ]; then
        echo "[TRAP] Syncing remaining artifacts to BeeGFS..."
        rsync -av "${LOCAL_SCRATCH}/output/" "${OUTPUT_DIR}/" || true
    fi
    rm -rf "${LOCAL_SCRATCH}" || true
    echo "[TRAP] Cleanup complete."
    exit ${exit_code}
}
trap cleanup_scratch EXIT TERM INT HUP USR1

# 4. Sequential High-Speed Stage-In Protocol
echo "[STAGE-IN] Streaming uncompressed national archive from BeeGFS to node scratch..."
t_stage_start=$(date +%s)
tar -C "${LOCAL_SCRATCH}/data" -xf "${DATA_TAR}"
t_stage_elapsed=$(( $(date +%s) - t_stage_start ))
echo "[STAGE-IN] Completed 34.8 GB transfer in ${t_stage_elapsed} seconds."

# 5. Execute Training via Apptainer with NUMA Interleaving
echo "[EXECUTION] Launching Model M5 Training via Apptainer..."
apptainer exec \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${LOCAL_SCRATCH}:${LOCAL_SCRATCH}" \
    --bind "${LOCAL_SCRATCH}:/tmp" \
    "${CONTAINER_SIF}" \
    numactl --interleave=all python "${PROJECT_DIR}/train_rf.py" \
        --model m5 \
        --mode cv \
        --scope national \
        --features-dir "${LOCAL_SCRATCH}/data" \
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

# 6. Stage-Out Protocol: Copy Artifacts to BeeGFS
echo "[STAGE-OUT] Syncing model checkpoints and metrics to BeeGFS..."
rsync -av "${LOCAL_SCRATCH}/output/" "${OUTPUT_DIR}/"

echo "=== [JOB COMPLETE] Finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
