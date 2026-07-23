#!/usr/bin/env bash
# Score every run under output/pred/{family}/{run}/ (test split by default).
# Finds nested runs (pred/d/..., pred/g/..., pred/d_old_normal/..., …).
# Skip if scores/eval_metrics.json exists unless FORCE=1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-/home/kasina/lab/3dmaterials/2d_3d_pipeline/2D_3D_pipeline/pipeline/data/smudgeremoval/finaldataset}"
SPLIT_DIR="${SPLIT_DIR:-${DATA_DIR}/splits/seed42_tvt_70_15_15}"
PRED_ROOT="${PRED_ROOT:-${ROOT}/output/pred}"
FORCE="${FORCE:-0}"

if [[ ! -d "$PRED_ROOT" ]]; then
  echo "No pred root: $PRED_ROOT" >&2
  exit 1
fi

echo "DATA_DIR=$DATA_DIR"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "PRED_ROOT=$PRED_ROOT"
echo "FORCE=$FORCE"
echo

# Any directory that already has normal/*.npy is a run to score
mapfile -t runs < <(
  find "$PRED_ROOT" -type d -name normal | sort | while read -r nd; do
    dirname "$nd"
  done
)

if [[ ${#runs[@]} -eq 0 ]]; then
  echo "No pred runs with normal/ under $PRED_ROOT" >&2
  exit 1
fi

for run_dir in "${runs[@]}"; do
  rel="${run_dir#"$PRED_ROOT"/}"

  n_npy="$(find "$run_dir/normal" -maxdepth 1 -type f -name '*.npy' | wc -l)"
  if [[ "$n_npy" -eq 0 ]]; then
    echo "Skip $rel (empty normal/)"
    continue
  fi

  out_dir="$run_dir/scores"
  metrics_json="$out_dir/eval_metrics.json"
  if [[ -f "$metrics_json" && "$FORCE" != "1" ]]; then
    echo "Skip $rel (already scored: $metrics_json)"
    continue
  fi

  echo "=== Scoring $rel → $out_dir ==="
  python score_finaldataset.py \
    --data_dir "$DATA_DIR" \
    --prediction_dir "$run_dir" \
    --output_dir "$out_dir" \
    --split_dir "$SPLIT_DIR"
  echo
done

echo "Done."
