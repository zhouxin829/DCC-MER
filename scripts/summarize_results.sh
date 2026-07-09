#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

SUMMARY_TYPE="${1:-5run}"
RUN_ROOT="${2:-runs}"
PYTHON="${PYTHON:-python}"

PROJECT_DIR="$(resolve_project_dir)"
cd "$PROJECT_DIR"

case "$SUMMARY_TYPE" in
  5run|main)
    require_project_file "$PROJECT_DIR" "summarize_dcc_all_5runs.py"
    "$PYTHON" summarize_dcc_all_5runs.py "$RUN_ROOT"
    ;;
  ablation|8ablation)
    require_project_file "$PROJECT_DIR" "collect_dcc_8ablation_results.py"
    "$PYTHON" collect_dcc_8ablation_results.py "$RUN_ROOT"
    ;;
  robustness)
    require_project_file "$PROJECT_DIR" "dcc_multiseed_tools/summarize_robustness_retrain.py"
    "$PYTHON" dcc_multiseed_tools/summarize_robustness_retrain.py --root "$RUN_ROOT"
    ;;
  mechanism)
    require_project_file "$PROJECT_DIR" "dcc_multiseed_tools/summarize_mechanism_controls.py"
    "$PYTHON" dcc_multiseed_tools/summarize_mechanism_controls.py --root "$RUN_ROOT"
    ;;
  diagnostics)
    require_project_file "$PROJECT_DIR" "dcc_multiseed_tools/summarize_retrain_diagnostics.py"
    "$PYTHON" dcc_multiseed_tools/summarize_retrain_diagnostics.py --root "$RUN_ROOT"
    ;;
  metrics)
    "$PYTHON" "$SCRIPT_DIR/evaluate_saved_metrics.py" "$RUN_ROOT"
    ;;
  *)
    echo "Unknown summary type: $SUMMARY_TYPE" >&2
    echo "Valid types: 5run, ablation, robustness, mechanism, diagnostics, metrics" >&2
    exit 2
    ;;
esac

