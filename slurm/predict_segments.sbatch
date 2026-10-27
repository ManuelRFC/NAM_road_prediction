#!/bin/bash
# ==============================================================================
# SLURM BATCH SCRIPT: NATIONAL SEGMENT-LEVEL INFERENCE & AGGREGATION
# CLUSTER: FUCHS-CSC (Goethe University Frankfurt)
# ==============================================================================
#SBATCH --job-name=rf_m5_predict_segments
#SBATCH --partition=fuchs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/fuchs/agmisc/cmiel/road_ml/logs/predict_segments_%j.out
#SBATCH --error=/work/fuchs/agmisc/cmiel/road_ml/logs/predict_segments_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cmiel@em.uni-frankfurt.de

set -euo pipefail

echo "=== [JOB START] FUCHS-CSC SLURM ID: ${SLURM_JOB_ID} ==="
echo "Node: $(hostname) | Allocated Cores: ${SLURM_CPUS_PER_TASK}"
echo "Start Time: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# 1. Thread Management: Prevent 400-thread CPU thrashing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# 2. Filesystem & Storage Hierarchy Setup
CODE_DIR="${HOME}/road_ml"
WORK_DIR="/work/fuchs/agmisc/cmiel/road_ml"
DATA_INPUT="${WORK_DIR}/data/parquet_national"
ROADS_SHP="${WORK_DIR}/data/shapefiles/DistrictRoads_with training data.shp"

# Default to latest completed production model (Job 1801371), or pass via environment
MODEL_PATH="${MODEL_PATH:-${WORK_DIR}/results/m5_monolithic_1801371/models/RF_M5_national_full.joblib}"
OUTPUT_CSV="${OUTPUT_CSV:-${WORK_DIR}/results/m5_monolithic_1801371/namibia_national_segment_predictions_m5.csv}"

LOCAL_SCRATCH="/localscratch/${SLURM_JOB_ID}"
if [ ! -d "/localscratch" ]; then
    LOCAL_SCRATCH="/tmp/${SLURM_JOB_ID}"
fi
export TMPDIR="${LOCAL_SCRATCH}"
export JOBLIB_TEMP_FOLDER="${LOCAL_SCRATCH}/joblib_cache"

mkdir -p "${JOBLIB_TEMP_FOLDER}"
mkdir -p "${WORK_DIR}/logs"
mkdir -p "$(dirname "${OUTPUT_CSV}")"

# 3. Scratch Cleanup Trap
cleanup_scratch() {
    local exit_code=$?
    echo "[TRAP] Job exiting with code ${exit_code}. Purging local scratch: ${LOCAL_SCRATCH}"
    rm -rf "${LOCAL_SCRATCH}" || true
    echo "[TRAP] Cleanup complete."
    exit ${exit_code}
}
trap cleanup_scratch EXIT TERM INT HUP USR1

# 4. Environment Activation
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

if [ -f "${HOME}/my_environments/geo_env/bin/activate" ]; then
    echo "[ENV] Activating existing geo_env..."
    source "${HOME}/my_environments/geo_env/bin/activate"
elif [ -f "${WORK_DIR}/venv/bin/activate" ]; then
    echo "[ENV] Activating project venv..."
    source "${WORK_DIR}/venv/bin/activate"
fi

# 5. Execute Streaming Batch Inference & Segment Aggregation
echo "[EXECUTION] Launching Segment Prediction & Aggregation..."
python "${CODE_DIR}/predict_segments.py" \
    --model-path "${MODEL_PATH}" \
    --features-dir "${DATA_INPUT}" \
    --roads-shp "${ROADS_SHP}" \
    --output-csv "${OUTPUT_CSV}" \
    --batch-size 250000 \
    --n-jobs ${SLURM_CPUS_PER_TASK}

echo "=== [JOB COMPLETE] Finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
echo "Output Summary File:"
ls -lh "${OUTPUT_CSV}"
