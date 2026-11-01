#!/bin/bash
# ==============================================================================
# SLURM ARRAY SCRIPT: MODEL SPECIFICATION BENCHMARK (7 MODELS ACROSS 5 SEEDS)
# CLUSTER: FUCHS-CSC (Goethe University Frankfurt)
# ==============================================================================
#SBATCH --job-name=rf_bench_7m
#SBATCH --partition=fuchs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=1-7
#SBATCH --output=/work/fuchs/agmisc/cmiel/road_ml/logs/benchmark_%A_%a.out
#SBATCH --error=/work/fuchs/agmisc/cmiel/road_ml/logs/benchmark_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cmiel@em.uni-frankfurt.de

set -euo pipefail

# 1. Map Slurm Array Index (1-7) to Model
MODELS=(m1 m2 m3 m4 m5 m6 m7)
MODEL_IDX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL=${MODELS[$MODEL_IDX]}

SEEDS=(42 101 202 303 404)

echo "=== [JOB START] FUCHS-CSC BENCHMARK TASK: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname) | Allocated Cores: ${SLURM_CPUS_PER_TASK}"
echo "Model: ${MODEL^^} | Evaluating 5 Stratified Seeds: ${SEEDS[*]}"
echo "Start Time: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# 2. Thread Management: Prevent CPU thread thrashing
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
SPLITS_FILE="${CODE_DIR}/splits/master_cv_splits_5seeds.csv"
OUTPUT_DIR="${WORK_DIR}/results/benchmark_m1_m7"

LOCAL_SCRATCH="/localscratch/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ ! -d "/localscratch" ]; then
    LOCAL_SCRATCH="/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
export TMPDIR="${LOCAL_SCRATCH}"
export JOBLIB_TEMP_FOLDER="${LOCAL_SCRATCH}/joblib_cache"

mkdir -p "${JOBLIB_TEMP_FOLDER}"
mkdir -p "${OUTPUT_DIR}/metrics"
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

# 6. Execute Model Cross-Validation across all 5 Seeds
for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "[EXECUTION] Model: ${MODEL^^} | Seed: ${SEED} | Time: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "----------------------------------------------------------------------"

    python "${CODE_DIR}/train_rf.py" \
        --model "${MODEL}" \
        --mode cv \
        --scope national \
        --features-dir "${DATA_INPUT}" \
        --roads-shp "${ROADS_SHP}" \
        --splits-file "${SPLITS_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --scratch-dir "${LOCAL_SCRATCH}" \
        --seed "${SEED}" \
        --n-trees 300 \
        --max-depth 25 \
        --min-samples-split 10 \
        --min-samples-leaf 4 \
        --criterion entropy \
        --class-weight balanced \
        --n-jobs ${SLURM_CPUS_PER_TASK} \
        --backend threading

    echo "[SEED COMPLETE] Model ${MODEL^^} with Seed ${SEED} finished successfully."
done

echo "=== [TASK COMPLETE] Finished All 5 Seeds for Model ${MODEL^^}: $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
