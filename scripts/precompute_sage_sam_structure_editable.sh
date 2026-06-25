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
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  OMP_NUM_THREADS=4
fi
if [[ -z "${MKL_NUM_THREADS:-}" || ! "${MKL_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi

DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-260513_data_multiclass}"
OUTPUT_CACHE="${OUTPUT_CACHE:-./structure_cache/${DATASET}}"

SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"
MODEL_TYPE="${MODEL_TYPE:-vit_b}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-3}"
IN_CHANNELS="${IN_CHANNELS:-3}"
POINT_NUMS="${POINT_NUMS:-5}"
BOX_NUMS="${BOX_NUMS:-1}"
MOD="${MOD:-seg}"
THD="${THD:-0}"
CHUNK="${CHUNK:-1}"
STRUCTURE_GRID_SIZE="${STRUCTURE_GRID_SIZE:-32}"

# Set SYNTHETIC=1 only for code smoke tests. Formal training should use a real SAM checkpoint.
SYNTHETIC="${SYNTHETIC:-0}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export MKL_NUM_THREADS

cmd=(
  "${PYTHON_BIN}" tools/precompute_sam_structure.py
  --data_path "${DATA_PATH}"
  --dataset "${DATASET}"
  --output_cache "${OUTPUT_CACHE}"
  --model_type "${MODEL_TYPE}"
  --device "${DEVICE}"
  --image_size "${IMAGE_SIZE}"
  --num_classes "${NUM_CLASSES}"
  --in_channels "${IN_CHANNELS}"
  --point_nums "${POINT_NUMS}"
  --box_nums "${BOX_NUMS}"
  --mod "${MOD}"
  --chunk "${CHUNK}"
  --structure_grid_size "${STRUCTURE_GRID_SIZE}"
)

if [[ "${THD}" == "1" ]]; then
  cmd+=(--thd)
fi

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
echo "  NUM_CLASSES=${NUM_CLASSES}"
echo "  IN_CHANNELS=${IN_CHANNELS}"
echo "  MOD=${MOD}"
echo "  STRUCTURE_GRID_SIZE=${STRUCTURE_GRID_SIZE}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo
printf 'Command:'
printf ' %q' "${cmd[@]}"
echo
echo

exec "${cmd[@]}"
