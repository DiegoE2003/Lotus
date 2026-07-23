#!/usr/bin/env bash
# Infer every FT checkpoint under output/model/{family}/{run}/ that lacks preds.
# Layout (keep family folders):
#   model/d/ftlotus_d_2bk_16bs_1kstp  →  pred/d/d_2bk_16bs_1kstp
#   model/g/2bk_16bs_4kstp            →  pred/g/2bk_16bs_4kstp
#   model/d_old_normal/2bk_16bs_1kstp →  pred/d_old_normal/2bk_16bs_1kstp
# Skip if pred/.../normal/*.npy already exist unless FORCE=1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

INPUT_DIR="${INPUT_DIR:-/home/kasina/lab/3dmaterials/2d_3d_pipeline/2D_3D_pipeline/pipeline/data/smudgeremoval/finaldataset_rgb_flat}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT}/output/model}"
PRED_ROOT="${PRED_ROOT:-${ROOT}/output/pred}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FORCE="${FORCE:-0}"
SEED="${SEED:-42}"

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "No model root: $MODEL_ROOT" >&2
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "No input dir (flat RGB): $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$PRED_ROOT"

echo "INPUT_DIR=$INPUT_DIR"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "PRED_ROOT=$PRED_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "FORCE=$FORCE"
echo

mapfile -t index_files < <(find "$MODEL_ROOT" -type f -name 'model_index.json' | sort)
if [[ ${#index_files[@]} -eq 0 ]]; then
  echo "No model_index.json under $MODEL_ROOT" >&2
  exit 1
fi

for index_json in "${index_files[@]}"; do
  model_dir="$(dirname "$index_json")"
  rel="${model_dir#"$MODEL_ROOT"/}"

  # Expect model/{family}/{run}/model_index.json (ignore deeper checkpoint copies)
  family="$(dirname "$rel")"
  run="$(basename "$rel")"
  if [[ "$family" == "." ]]; then
    echo "Skip $rel (expected model/<family>/<run>/)"
    continue
  fi

  # Match existing pred naming: strip leading ftlotus_ from run folder only
  run_pred="${run#ftlotus_}"
  out_dir="$PRED_ROOT/$family/$run_pred"
  normal_dir="$out_dir/normal"

  n_existing=0
  if [[ -d "$normal_dir" ]]; then
    n_existing="$(find "$normal_dir" -maxdepth 1 -type f -name '*.npy' | wc -l)"
  fi
  if [[ "$n_existing" -gt 0 && "$FORCE" != "1" ]]; then
    echo "Skip $rel → pred/$family/$run_pred (already has $n_existing npy)"
    continue
  fi

  class_name="$(python -c "import json; print(json.load(open('$index_json')).get('_class_name',''))")"
  case "$class_name" in
    LotusGPipeline) mode=generation ;;
    LotusDPipeline) mode=regression ;;
    *)
      echo "Skip $rel (unknown _class_name='$class_name')"
      continue
      ;;
  esac

  mkdir -p "$(dirname "$out_dir")"
  echo "=== Infer $rel ($class_name / $mode) → pred/$family/$run_pred ==="
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python infer.py \
    --pretrained_model_name_or_path "$model_dir" \
    --prediction_type sample \
    --task_name normal \
    --mode "$mode" \
    --half_precision \
    --seed "$SEED" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$out_dir"
  echo
done

echo "Done."
