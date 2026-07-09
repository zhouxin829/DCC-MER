#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DATA_PATH="${1:-${DATA_PATH:-./MSA_Datasets}}"
RUN_ROOT="${2:-${RUN_ROOT:-runs/dcc_all_5runs}}"
GPU_ID="${3:-${GPU_ID:-0}}"
DATASETS_CSV="${4:-${DATASETS:-SIMS,MOSI,MOSEI}}"
SEEDS_CSV="${SEEDS:-3328683074,4136559363,1686802513,2124692648,964165003}"
PYTHON="${PYTHON:-python}"

PROJECT_DIR="$(resolve_project_dir)"
require_project_file "$PROJECT_DIR" "dcc_main.py"

csv_to_array "$DATASETS_CSV" DATASETS_ARR
csv_to_array "$SEEDS_CSV" SEEDS_ARR

echo "[DCC-MER] Multi-seed run root: $RUN_ROOT"
echo "[DCC-MER] Datasets: ${DATASETS_ARR[*]}"
echo "[DCC-MER] Seeds: ${SEEDS_ARR[*]}"

for dataset in "${DATASETS_ARR[@]}"; do
  run_idx=1
  for seed in "${SEEDS_ARR[@]}"; do
    run_name="$(printf "run_%02d" "$run_idx")"
    run_dir="$RUN_ROOT/$dataset/$run_name"
    echo "[DCC-MER] Running $dataset $run_name seed=$seed"
    PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" bash "$SCRIPT_DIR/train_dcc_mer.sh" \
      "$dataset" "$DATA_PATH" "$run_dir" "$GPU_ID" "$seed"
    run_idx=$((run_idx + 1))
  done
done

if [[ -f "$PROJECT_DIR/summarize_dcc_all_5runs.py" ]]; then
  cd "$PROJECT_DIR"
  "$PYTHON" summarize_dcc_all_5runs.py "$RUN_ROOT"
fi

echo "[DCC-MER] Multi-seed runs completed."

