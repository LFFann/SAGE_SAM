#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SAGE-SAM editable training launcher
# Edit the variables in this section for normal experiments.
# You can also override any variable from the command line, for example:
#   CUDA_VISIBLE_DEVICES=1 MAX_ITERATIONS=30000 bash scripts/train_sage_sam_editable.sh
# ============================================================

# Runtime
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
USE_AMP="${USE_AMP:-1}"
SEED="${SEED:-2026}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Dataset
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-260513_data_multiclass}"
NUM_CLASSES="${NUM_CLASSES:-3}"
IN_CHANNELS="${IN_CHANNELS:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

# SAM structure cache
STRUCTURE_CACHE="${STRUCTURE_CACHE:-./structure_cache/${DATASET}}"
STRUCTURE_CACHE_MODE="${STRUCTURE_CACHE_MODE:-required}"
STRUCTURE_GRID_SIZE="${STRUCTURE_GRID_SIZE:-32}"
STRUCTURE_TARGET_PRECISION="${STRUCTURE_TARGET_PRECISION:-0.9}"

# Calibration
CALIBRATION_RATIO="${CALIBRATION_RATIO:-0.2}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-4}"
CALIBRATION_MODE="${CALIBRATION_MODE:-strict_once}"
CALIBRATION_INTERVAL="${CALIBRATION_INTERVAL:-500}"
CONFORMAL_ALPHA="${CONFORMAL_ALPHA:-0.1}"
CALIBRATION_MANIFEST="${CALIBRATION_MANIFEST:-}"
GROUP_REGEX="${GROUP_REGEX:-}"
STRUCTURE_CALIBRATION="${STRUCTURE_CALIBRATION:-}"

# Training schedule
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-1000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-250}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"

# Batch sizes
LABELED_BATCH_SIZE="${LABELED_BATCH_SIZE:-8}"
UNLABELED_BATCH_SIZE="${UNLABELED_BATCH_SIZE:-8}"

# Optimizer
OPTIMIZER="${OPTIMIZER:-adamw}"
UNET_LR="${UNET_LR:-0.001}"
VNET_LR="${VNET_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MOMENTUM="${MOMENTUM:-0.9}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-0.0}"

# Output and checkpoint
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-SAGE_SAM_3Class}"
RESUME="${RESUME:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"
ABLATION_SUPERVISED_ONLY="${ABLATION_SUPERVISED_ONLY:-0}"

export CUDA_VISIBLE_DEVICES

cmd=(
  "${PYTHON_BIN}" train_sage_sam.py
  --data_path "${DATA_PATH}"
  --dataset "${DATASET}"
  --structure_cache "${STRUCTURE_CACHE}"
  --structure_cache_mode "${STRUCTURE_CACHE_MODE}"
  --structure_grid_size "${STRUCTURE_GRID_SIZE}"
  --structure_target_precision "${STRUCTURE_TARGET_PRECISION}"
  --output_dir "${OUTPUT_DIR}"
  --experiment_name "${EXPERIMENT_NAME}"
  --num_classes "${NUM_CLASSES}"
  --in_channels "${IN_CHANNELS}"
  --image_size "${IMAGE_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --labeled_batch_size "${LABELED_BATCH_SIZE}"
  --unlabeled_batch_size "${UNLABELED_BATCH_SIZE}"
  --calibration_batch_size "${CALIBRATION_BATCH_SIZE}"
  --calibration_ratio "${CALIBRATION_RATIO}"
  --calibration_mode "${CALIBRATION_MODE}"
  --calibration_interval "${CALIBRATION_INTERVAL}"
  --conformal_alpha "${CONFORMAL_ALPHA}"
  --max_iterations "${MAX_ITERATIONS}"
  --warmup_iterations "${WARMUP_ITERATIONS}"
  --validation_interval "${VALIDATION_INTERVAL}"
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --optimizer "${OPTIMIZER}"
  --unet_lr "${UNET_LR}"
  --vnet_lr "${VNET_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --momentum "${MOMENTUM}"
  --gradient_clip_norm "${GRADIENT_CLIP_NORM}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ "${USE_AMP}" == "1" ]]; then
  cmd+=(--amp)
else
  cmd+=(--no_amp)
fi

if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  cmd+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi

if [[ -n "${CALIBRATION_MANIFEST}" ]]; then
  cmd+=(--calibration_manifest "${CALIBRATION_MANIFEST}")
fi

if [[ -n "${GROUP_REGEX}" ]]; then
  cmd+=(--group_regex "${GROUP_REGEX}")
fi

if [[ -n "${STRUCTURE_CALIBRATION}" ]]; then
  cmd+=(--structure_calibration "${STRUCTURE_CALIBRATION}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry_run)
fi

if [[ "${ABLATION_SUPERVISED_ONLY}" == "1" ]]; then
  cmd+=(--ablation_supervised_only)
fi

cmd+=("$@")

echo "Starting SAGE-SAM training"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  STRUCTURE_CACHE=${STRUCTURE_CACHE}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  WARMUP_ITERATIONS=${WARMUP_ITERATIONS}"
echo "  LABELED_BATCH_SIZE=${LABELED_BATCH_SIZE}"
echo "  UNLABELED_BATCH_SIZE=${UNLABELED_BATCH_SIZE}"
echo
printf 'Command:'
printf ' %q' "${cmd[@]}"
echo
echo

exec "${cmd[@]}"
