#!/usr/bin/env bash
# Evaluate Lotus Normal G on DiLiGenT-Pi (MAE / MAngE in degrees).
#
# 1. Download DiLiGenT-Pi from https://photometricstereo.github.io/diligentpi.html
# 2. Set DATA_DIR to the extracted root (or pmsData parent).
# 3. Run:
#      bash eval_scripts/eval-normal-g-diligentpi.sh

set -euo pipefail

export CUDA="${CUDA:-0}"
export DATA_DIR="${DATA_DIR:-/data/DiLiGent-Pi/DiLiGenT-Pi_release}"
export GT_DIR="${GT_DIR:-}"   # set if Normal_gt is in a separate evaluation download
export OUTPUT_DIR="${OUTPUT_DIR:-output/DiLiGenT-Pi_LotusG}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-jingheya/lotus-normal-g-v1-1}"
export LIGHT_INDEX="${LIGHT_INDEX:-0}"

cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ -n "$GT_DIR" ]]; then
  EXTRA_ARGS+=(--gt_dir "$GT_DIR")
fi

CUDA_VISIBLE_DEVICES="$CUDA" python eval_diligentpi.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}" \
  --pretrained_model_name_or_path "$CHECKPOINT_DIR" \
  --mode generation \
  --half_precision \
  --seed 42 \
  --light_index "$LIGHT_INDEX" \
  --save_vis
