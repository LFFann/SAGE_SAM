#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SAGE-SAM editable test launcher
# Edit variables here, or override them from the command line:
#   CHECKPOINT=./outputs/SAGE_SAM_3Class/checkpoints/final.pth bash scripts/test_sage_sam_editable.sh
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  OMP_NUM_THREADS=4
fi
if [[ -z "${MKL_NUM_THREADS:-}" || ! "${MKL_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi

DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-260513_data_multiclass}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-SAGE_SAM_3Class}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${EXPERIMENT_NAME}/checkpoints/best_fused_dice.pth}"
SPLIT="${SPLIT:-test}"
METRICS_FILE="${METRICS_FILE:-${OUTPUT_DIR}/${EXPERIMENT_NAME}/metrics/${SPLIT}_metrics.json}"

NUM_CLASSES="${NUM_CLASSES:-3}"
IN_CHANNELS="${IN_CHANNELS:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export MKL_NUM_THREADS

cmd=(
  "${PYTHON_BIN}" test_sage_sam.py
  --data_path "${DATA_PATH}"
  --dataset "${DATASET}"
  --checkpoint "${CHECKPOINT}"
  --split "${SPLIT}"
  --output_dir "${OUTPUT_DIR}"
  --experiment_name "${EXPERIMENT_NAME}"
  --metrics_file "${METRICS_FILE}"
  --num_classes "${NUM_CLASSES}"
  --in_channels "${IN_CHANNELS}"
  --image_size "${IMAGE_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
)

cmd+=("$@")

echo "Starting SAGE-SAM test"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  CHECKPOINT=${CHECKPOINT}"
echo "  SPLIT=${SPLIT}"
echo "  METRICS_FILE=${METRICS_FILE}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo
printf 'Command:'
printf ' %q' "${cmd[@]}"
echo
echo

exec "${cmd[@]}"
