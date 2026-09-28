#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DRAFT_KV_PYTHON="${DRAFT_KV_PYTHON:-python3}"
HF_ROOT="${HF_HOME:-/workspace/huggingface}"
GPU_ID="${DRAFT_KV_GPU_ID:-0}"

if [[ ! -x "$DRAFT_KV_PYTHON" ]]; then
  echo "Missing Draft-KV Python: $DRAFT_KV_PYTHON" >&2
  exit 2
fi

forwarded_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-id)
      if [[ $# -lt 2 ]]; then
        echo "--gpu-id requires a physical GPU index" >&2
        exit 2
      fi
      GPU_ID="$2"
      shift 2
      ;;
    --gpu-id=*)
      GPU_ID="${1#*=}"
      shift
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Invalid GPU id: $GPU_ID" >&2
  exit 2
fi

echo "Running downstream evaluation on physical GPU $GPU_ID."

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="$HF_ROOT"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_ROOT/datasets}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
exec "$DRAFT_KV_PYTHON" script/draft_kv/run_stage3_eval.py \
  --device cuda:0 \
  "${forwarded_args[@]}"
