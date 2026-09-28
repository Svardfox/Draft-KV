#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export DRAFT_KV_GPU_ID="${DRAFT_KV_GPU_ID:-0}"
export DRAFT_KV_STAGE3_OUTPUT_DIR="${DRAFT_KV_STAGE3_OUTPUT_DIR:-/workspace/draft-kv/stage3_mc_training/train}"
train_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-id)
      [[ $# -ge 2 ]] || { echo "--gpu-id requires a value" >&2; exit 2; }
      export DRAFT_KV_GPU_ID="$2"; shift 2 ;;
    --gpu-id=*) export DRAFT_KV_GPU_ID="${1#*=}"; shift ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      export DRAFT_KV_STAGE3_OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*) export DRAFT_KV_STAGE3_OUTPUT_DIR="${1#*=}"; shift ;;
    -h|--help)
      echo "Usage: bash script/draft_kv/run_stage3_train_and_eval.sh [--gpu-id N] [--output-dir PATH] [trainer args]"
      echo "Train Stage 3, then evaluate its best.pt on all seven MC benchmarks."
      echo "Defaults: protection-weight=0.1, protection-tolerance=0.1; weight=0 gives CE ablation."
      echo "DRAFT_KV_DATASETS selects benchmarks; SMOKE=1 limits evaluation only."
      exit 0 ;;
    *) train_args+=("$1"); shift ;;
  esac
done
[[ "$DRAFT_KV_GPU_ID" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id" >&2; exit 2; }
cd "$REPO_ROOT"
export DRAFT_KV_STAGE3_OUTPUT_DIR="$(realpath -m "$DRAFT_KV_STAGE3_OUTPUT_DIR")"
export DRAFT_KV_STAGE3_EVAL_LOG_ROOT="${DRAFT_KV_STAGE3_EVAL_LOG_ROOT:-$DRAFT_KV_STAGE3_OUTPUT_DIR/../eval_logs}"
for artifact in best.pt last.pt train_history.json train_result.json; do
  [[ ! -e "$DRAFT_KV_STAGE3_OUTPUT_DIR/$artifact" ]] || {
    echo "Training output exists; use a new --output-dir: $DRAFT_KV_STAGE3_OUTPUT_DIR" >&2
    exit 2
  }
done
mkdir -p "$DRAFT_KV_STAGE3_OUTPUT_DIR/logs"

bash script/draft_kv/run_stage3_training.sh "${train_args[@]}" \
  2>&1 | tee "$DRAFT_KV_STAGE3_OUTPUT_DIR/logs/training.log"

# Bind evaluation to this successful training run, never an inherited checkpoint.
export DRAFT_KV_CHECKPOINT="$DRAFT_KV_STAGE3_OUTPUT_DIR/best.pt"
[[ -f "$DRAFT_KV_CHECKPOINT" ]] || { echo "Training produced no best.pt" >&2; exit 1; }
bash script/draft_kv/run_stage3_eval_all.sh
echo "Completed training and benchmark evaluation: $DRAFT_KV_STAGE3_EVAL_LOG_ROOT/summary.txt"
