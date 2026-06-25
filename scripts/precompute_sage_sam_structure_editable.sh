#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SAGE-SAM editable SAM structure-cache precompute launcher
# Run this once before semi-supervised training.
# Example:
#   DATA_PATH=/root/autodl-tmp/echoData DATASET=260513_data_labeled30pct bash scripts/precompute_sage_sam_structure_editable.sh
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"

DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-260513_data_multiclass}"
OUTPUT_CACHE="${OUTPUT_CACHE:-./structure_cache/${DATASET}}"

SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"
MODEL_TYPE="${MODEL_TYPE:-vit_b}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
STRUCTURE_GRID_SIZE="${STRUCTURE_GRID_SIZE:-32}"

# Set SYNTHETIC=1 only for code smoke tests. Formal training should use a real SAM checkpoint.
SYNTHETIC="${SYNTHETIC:-0}"

export CUDA_VISIBLE_DEVICES

cmd=(
  "${PYTHON_BIN}" tools/precompute_sam_structure.py
  --data_path "${DATA_PATH}"
  --dataset "${DATASET}"
  --output_cache "${OUTPUT_CACHE}"
  --model_type "${MODEL_TYPE}"
  --device "${DEVICE}"
  --image_size "${IMAGE_SIZE}"
  --structure_grid_size "${STRUCTURE_GRID_SIZE}"
)

if [[ "${SYNTHETIC}" == "1" ]]; then
  cmd+=(--synthetic)
else
  cmd+=(--sam_checkpoint "${SAM_CHECKPOINT}")
fi

cmd+=("$@")

echo "Starting SAGE-SAM structure-cache precompute"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  OUTPUT_CACHE=${OUTPUT_CACHE}"
echo "  SAM_CHECKPOINT=${SAM_CHECKPOINT}"
echo "  MODEL_TYPE=${MODEL_TYPE}"
echo "  IMAGE_SIZE=${IMAGE_SIZE}"
echo "  STRUCTURE_GRID_SIZE=${STRUCTURE_GRID_SIZE}"
echo
printf 'Command:'
printf ' %q' "${cmd[@]}"
echo
echo

exec "${cmd[@]}"
