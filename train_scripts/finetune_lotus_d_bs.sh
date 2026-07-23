#!/usr/bin/env bash
# Fine-tune last UNet layers of Lotus D on TRAIN split only (shared stratified split).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/home/kasina/lab/3dmaterials/2d_3d_pipeline/2D_3D_pipeline/pipeline/data/smudgeremoval/finaldataset}"
export SPLIT_DIR="${SPLIT_DIR:-${TRAIN_DATA_DIR}/splits/seed42_tvt_70_15_15}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/output/model/d/1024res_2bk_4bs_1kstp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ ! -f "${SPLIT_DIR}/train_stems.txt" ]]; then
  echo "Creating shared train/val/test split at ${SPLIT_DIR}"
  python make_finaldataset_split.py \
    --data_dir "$TRAIN_DATA_DIR" \
    --output_dir "$SPLIT_DIR" \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
fi

python finetune_lotus_last_layers.py \
  --mode regression \
  --pretrained_model_name_or_path jingheya/lotus-normal-d-v1-1 \
  --train_data_dir "$TRAIN_DATA_DIR" \
  --split_dir "$SPLIT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --resolution 1024 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --random_flip \
  --max_train_steps 1000 \
  --learning_rate 3e-5 \
  --trainable_scope last_up_blocks \
  --train_last_n_up_blocks 2 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --allow_tf32 \
  --checkpointing_steps 500 \
  --validation_steps "${VALIDATION_STEPS:-200}" \
  --seed 42
