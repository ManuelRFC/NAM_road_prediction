#!/bin/bash
# ==============================================================================
# SLURM ARRAY SCRIPT: MODEL M5 SAMPLE COMPLEXITY / LEARNING CURVE EXPERIMENT
# CLUSTER: FUCHS-CSC (Goethe University Frankfurt)
# ==============================================================================
#SBATCH --job-name=rf_m5_learning_curve
#SBATCH --partition=fuchs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --array=1-6
#SBATCH --output=/work/fuchs/agmisc/cmiel/road_ml/logs/learning_curve_%A_%a.out
#SBATCH --error=/work/fuchs/agmisc/cmiel/road_ml/logs/learning_curve_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cmiel@em.uni-frankfurt.de

set -euo pipefail

# 1. Map Slurm Array Index to Training Fraction
FRACTIONS=(0.05 0.10 0.25 0.50 0.75 1.00)
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
FRACTION=${FRACTIONS[$ARRAY_INDEX]}

if [ "$FRACTION" == "1.00" ]; then
    N_REPS=1
else
    N_REPS=5
fi

echo "=== [JOB START] FUCHS-CSC SLURM ARRAY ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname) | Allocated Cores: ${SLURM_CPUS_PER_TASK}"
echo "Fraction: ${FRACTION} | Subsampling Repetitions: ${N_REPS}"
echo "Start Time: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# 2. Thread Management: Prevent CPU thrashing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# 3. Storage & Directory Setup
CODE_DIR="${HOME}/road_ml"
WORK_DIR="/work/fuchs/agmisc/cmiel/road_ml"
DATA_INPUT="${WORK_DIR}/data/parquet_national"
ROADS_SHP="${WORK_DIR}/data/shapefiles/DistrictRoads_with training data.shp"
OUTPUT_DIR="${WORK_DIR}/results/learning_curve"
RAW_METRICS_DIR="${OUTPUT_DIR}/raw_tasks"
CACHE_FILE="${WORK_DIR}/data/cache_labeled_m5_dataset.parquet"

LOCAL_SCRATCH="/localscratch/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ ! -d "/localscratch" ]; then
    LOCAL_SCRATCH="/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
export TMPDIR="${LOCAL_SCRATCH}"
export JOBLIB_TEMP_FOLDER="${LOCAL_SCRATCH}/joblib_cache"

mkdir -p "${JOBLIB_TEMP_FOLDER}"
mkdir -p "${RAW_METRICS_DIR}"
mkdir -p "${WORK_DIR}/logs"

# 4. Scratch Cleanup Trap
cleanup_scratch() {
    local exit_code=$?
    echo "[TRAP] Task exiting with code ${exit_code}. Purging local scratch: ${LOCAL_SCRATCH}"
    rm -rf "${LOCAL_SCRATCH}" || true
    echo "[TRAP] Cleanup complete."
    exit ${exit_code}
}
trap cleanup_scratch EXIT TERM INT HUP USR1

# 5. Environment Activation
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

if [ -f "${HOME}/my_environments/geo_env/bin/activate" ]; then
    echo "[ENV] Activating existing geo_env..."
    source "${HOME}/my_environments/geo_env/bin/activate"
elif [ -f "${WORK_DIR}/venv/bin/activate" ]; then
    echo "[ENV] Activating project venv..."
    source "${WORK_DIR}/venv/bin/activate"
fi

# 6. Execute Learning Curve Task for this Fraction
echo "[EXECUTION] Launching Learning Curve Task (Fraction: ${FRACTION})..."
python "${CODE_DIR}/train_learning_curve.py" \
    --fraction "${FRACTION}" \
    --n-reps "${N_REPS}" \
    --features-dir "${DATA_INPUT}" \
    --roads-shp "${ROADS_SHP}" \
    --output-dir "${RAW_METRICS_DIR}" \
    --cache-file "${CACHE_FILE}" \
    --n-trees 100 \
    --n-jobs ${SLURM_CPUS_PER_TASK}

echo "=== [TASK COMPLETE] Finished Fraction ${FRACTION}: $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
