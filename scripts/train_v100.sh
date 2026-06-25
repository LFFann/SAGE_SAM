#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-260513_data_multiclass}"
STRUCTURE_CACHE="${STRUCTURE_CACHE:-./structure_cache/260513_data_multiclass}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-SAGE_SAM_3Class}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES

echo "Starting SAGE-SAM training:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  STRUCTURE_CACHE=${STRUCTURE_CACHE}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  EXPERIMENT_NAME=${EXPERIMENT_NAME}"

"${PYTHON_BIN}" train_sage_sam.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --structure_cache "${STRUCTURE_CACHE}" \
  --structure_cache_mode required \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --num_classes 3 \
  --in_channels 3 \
  --image_size 256 \
  --labeled_batch_size 8 \
  --unlabeled_batch_size 8 \
  --calibration_batch_size 4 \
  --max_iterations 10000 \
  --warmup_iterations 1000 \
  --validation_interval 250 \
  --checkpoint_interval 500 \
  --device cuda \
  --amp \
  "$@"
