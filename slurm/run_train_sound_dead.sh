#!/bin/bash
#SBATCH --job-name=KnotTrain
#SBATCH --output=train_sound_dead_%j.out
#SBATCH --error=train_sound_dead_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=48:00:00
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
[[ -f "scripts/train_sound_dead.py" ]] || { echo "Set REPO_DIR to the extracted repository." >&2; exit 1; }

"${PYTHON_BIN}" -u scripts/train_sound_dead.py \
    --block-root "${BLOCK_ROOT}" --index-csv "${INDEX_CSV}" \
    --output-dir "${OUTPUT_DIR}" --image-state "${IMAGE_STATE}" \
    --epochs 100 --early-stop-patience 15 --lr-patience 5 --lr-factor 0.5 \
    --learning-rate 0.0002 --weight-decay 0 --batch-size 32 \
    --evaluation-batch-size 64 --num-workers 4 --volume-cache-size 4 \
    --seed 42 --threshold 0.5 "$@"
