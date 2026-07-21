#!/usr/bin/env bash
# Fine-tune last UNet layers of Lotus G on TRAIN split only (shared stratified split).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/home/kasina/lab/3dmaterials/2d_3d_pipeline/2D_3D_pipeline/pipeline/data/smudgeremoval/finaldataset}"
export SPLIT_DIR="${SPLIT_DIR:-${TRAIN_DATA_DIR}/splits/seed42_val20}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/output/finetune_lotus_g_bs16_lr3e-5_2ksteps}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${SPLIT_DIR}/train_stems.txt" ]]; then
  echo "Creating shared split at ${SPLIT_DIR}"
  python make_finaldataset_split.py \
    --data_dir "$TRAIN_DATA_DIR" \
    --output_dir "$SPLIT_DIR" \
    --val_ratio 0.2 \
    --seed 42
fi

python finetune_lotus_last_layers.py \
  --mode generation \
  --pretrained_model_name_or_path jingheya/lotus-normal-g-v1-1 \
  --train_data_dir "$TRAIN_DATA_DIR" \
  --split_dir "$SPLIT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --resolution 576 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --random_flip \
  --max_train_steps 2000 \
  --learning_rate 3e-5 \
  --train_last_n_up_blocks 1 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --allow_tf32 \
  --checkpointing_steps 500 \
  --seed 42
