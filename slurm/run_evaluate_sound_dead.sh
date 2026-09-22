#!/bin/bash
#SBATCH --job-name=KnotEvaluate
#SBATCH --output=evaluate_sound_dead_%j.out
#SBATCH --error=evaluate_sound_dead_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_DIR="${PROJECT_DIR:-${REPO_DIR}/work}"
BLOCK_ROOT="${BLOCK_ROOT:-${PROJECT_DIR}/Individual_Knot_Crops_160x80x80_thesis_fit}"
INDEX_CSV="${INDEX_CSV:-${PROJECT_DIR}/Sound_Dead_Dataset_11x80x80_thesis_fit/patch_index_11x80x80_thesis_fit.csv}"
IMAGE_STATE="${IMAGE_STATE:-wet}"
RESULTS_NAME="${RESULTS_NAME:-results_${IMAGE_STATE}}"
OUTPUT_DIR="${PROJECT_DIR}/${RESULTS_NAME}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV:-WaiKnotCT}"
fi
cd "${REPO_DIR}"
export WAIKNOT_PROJECT_ROOT="${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
[[ -f "scripts/evaluate_sound_dead.py" ]] || { echo "Set REPO_DIR to the extracted repository." >&2; exit 1; }

EVALUATION_NAME="${EVALUATION_NAME:-evaluation}"
"${PYTHON_BIN}" -u scripts/evaluate_sound_dead.py \
    --block-root "${BLOCK_ROOT}" --index-csv "${INDEX_CSV}" \
    --checkpoint "${OUTPUT_DIR}/best_model.pt" \
    --output-dir "${OUTPUT_DIR}/${EVALUATION_NAME}" --image-state "${IMAGE_STATE}" \
    --threshold 0.5 --batch-size 64 --num-workers 4 --volume-cache-size 4 \
    --timing-repeats 3 --timing-warmup 20 --seed 42 "$@"
