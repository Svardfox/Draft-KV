#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DRAFT_KV_PYTHON="${DRAFT_KV_PYTHON:-python3}"
GPU_ID="${DRAFT_KV_GPU_ID:-0}"
HF_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
RUN_ROOT="${DRAFT_KV_RUN_ROOT:-$HOME/draft-kv-runs}"

MC_DATA_DIR="${DRAFT_KV_STAGE3_DATA_DIR:-$RUN_ROOT/stage3/data}"
INITIAL_STAGE2="${DRAFT_KV_STAGE3_INITIAL_STAGE2:-$HOME/checkpoints/openhermes_stage2/train/last.pt}"
RECON_DATA="${DRAFT_KV_STAGE3_RECON_DATA:-$HOME/data/openhermes_reconstruction}"
OUTPUT_DIR="${DRAFT_KV_STAGE3_OUTPUT_DIR:-$RUN_ROOT/stage3/train}"

usage() {
  cat <<'EOF'
Usage:
  run_stage3_training.sh [--gpu-id N] [extra trainer args]

Trains Stage 3 from an OpenHermes Stage-2 checkpoint using ARC option
logits, matched/deranged controls, and OpenHermes reconstruction replay.

Environment overrides:
  DRAFT_KV_PYTHON
  DRAFT_KV_STAGE3_DATA_DIR
  DRAFT_KV_STAGE3_INITIAL_STAGE2
  DRAFT_KV_STAGE3_RECON_DATA
  DRAFT_KV_STAGE3_OUTPUT_DIR
EOF
}

forwarded=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-id)
      [[ $# -ge 2 ]] || { echo "--gpu-id requires a value" >&2; exit 2; }
      GPU_ID="$2"
      shift 2
      ;;
    --gpu-id=*)
      GPU_ID="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: $GPU_ID" >&2; exit 2; }
command -v "$DRAFT_KV_PYTHON" >/dev/null 2>&1 || {
  echo "Missing Python executable: $DRAFT_KV_PYTHON" >&2
  exit 2
}

mkdir -p "$OUTPUT_DIR/logs"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="$HF_ROOT"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_ROOT/datasets}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

exec "$DRAFT_KV_PYTHON" script/draft_kv/train_stage3.py \
  --device cuda:0 \
  --data-dir "$MC_DATA_DIR" \
  --reconstruction-data-dir "$RECON_DATA" \
  --initial-stage2-checkpoint "$INITIAL_STAGE2" \
  --output-dir "$OUTPUT_DIR" \
  --protection-weight 0.1 \
  --protection-tolerance 0.1 \
  --mc-updates-per-replay 4 \
  "${forwarded[@]}"
