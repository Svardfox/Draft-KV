#!/usr/bin/env bash
# One-click seven-benchmark MC evaluation for the Stage 3 checkpoint:
#   Qwen3-0.6B -> Qwen2.5-0.5B-Instruct
#
# Usage:
#   nohup bash script/draft_kv/run_stage3_eval_all.sh \
#     > /tmp/eval_stage3_mc_all.log 2>&1 &
#
# Optional:
#   DRAFT_KV_GPU_ID=1 bash script/draft_kv/run_stage3_eval_all.sh
#   DRAFT_KV_DATASETS=gsm-mc bash script/draft_kv/run_stage3_eval_all.sh
#   SMOKE=1 bash script/draft_kv/run_stage3_eval_all.sh
#   DRAFT_KV_SKIP_STANDALONE=1 bash script/draft_kv/run_stage3_eval_all.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DRAFT_KV_PYTHON="${DRAFT_KV_PYTHON:-python3}"
GPU_ID="${DRAFT_KV_GPU_ID:-0}"
DRAFT_KV_ROOT="${DRAFT_KV_ROOT:-/workspace/draft-kv}"
MODEL_ROOT="${DRAFT_KV_MODEL_ROOT:-/workspace/models}"
DATA_ROOT="${DRAFT_KV_DATA_ROOT:-/workspace/datasets}"
MATCHED_OUTPUT_ROOT="${DRAFT_KV_MATCHED_OUTPUT_ROOT:-$DRAFT_KV_ROOT/downstream_mc}"
STANDALONE_OUTPUT_ROOT="${DRAFT_KV_STANDALONE_OUTPUT_ROOT:-$DRAFT_KV_ROOT/standalone_sharer_only}"
LOG_ROOT="${DRAFT_KV_STAGE3_EVAL_LOG_ROOT:-$DRAFT_KV_ROOT/logs/eval_stage3_mc_all}"
SMOKE="${SMOKE:-0}"
SKIP_STANDALONE="${DRAFT_KV_SKIP_STANDALONE:-0}"

SHARER="${DRAFT_KV_SHARER:-$MODEL_ROOT/Qwen3-0.6B}"
RECEIVER="${DRAFT_KV_RECEIVER:-$MODEL_ROOT/Qwen2.5-0.5B-Instruct}"
LAYER_MAPPING="${DRAFT_KV_LAYER_MAPPING:-14:18,16:20,18:22,20:24}"
CHECKPOINT="${DRAFT_KV_CHECKPOINT:-$DRAFT_KV_ROOT/stage3_mc_training/train/best.pt}"
# Space/comma separated receiver conditions.  Matched and Deranged are the
# only Stage 3-specific columns; pass "matched deranged" to skip the invariant
# Zero path.  The Sharer draft cache identity always covers all conditions,
# so narrowing this list never invalidates an existing cache.
EVAL_CONDITIONS="${DRAFT_KV_EVAL_CONDITIONS:-zero matched deranged}"

# Cheapest datasets first. Override with DRAFT_KV_DATASETS.
DATASETS="${DRAFT_KV_DATASETS:-openbookqa,arc-c,gsm-mc,ceval,arc-e,mmlu-redux,math-mc}"
IFS=',' read -r -a DATASET_LIST <<<"$DATASETS"

input_limit() {
  case "$1" in
    math-mc) echo 1536 ;;
    *) echo 1280 ;;
  esac
}

draft_limit() {
  case "$1" in
    math-mc) echo 1024 ;;
    *) echo 512 ;;
  esac
}

draft_batch() {
  local specific
  specific="DRAFT_KV_DRAFT_BATCH_SIZE_$(printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_')"
  if [[ -n "${!specific:-}" ]]; then
    echo "${!specific}"
    return
  fi
  if [[ -n "${DRAFT_KV_DRAFT_BATCH_SIZE:-}" ]]; then
    echo "$DRAFT_KV_DRAFT_BATCH_SIZE"
    return
  fi
  case "$1" in
    math-mc) echo 8 ;;
    mmlu-redux) echo 16 ;;
    *) echo 32 ;;
  esac
}

eval_batch() {
  case "$1" in
    math-mc) echo "${DRAFT_KV_EVAL_BATCH_SIZE:-24}" ;;
    *) echo "${DRAFT_KV_EVAL_BATCH_SIZE:-48}" ;;
  esac
}

adapted_file() {
  case "$1" in
    mmlu-redux) echo "$DATA_ROOT/mmlu-redux-2.0/adapted/test.jsonl" ;;
    arc-e|arc-c) echo "$DATA_ROOT/ai2_arc/adapted/$1/test.jsonl" ;;
    openbookqa) echo "$DATA_ROOT/openbookqa/adapted/test.jsonl" ;;
    ceval) echo "$DATA_ROOT/ceval/adapted/val.jsonl" ;;
    *) echo "$DATA_ROOT/mc-evaluation/$1/adapted/test.jsonl" ;;
  esac
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

# DRAFT_KV_PYTHON may be an executable path or a bare command name on PATH.
if [[ "$DRAFT_KV_PYTHON" != */* ]]; then
  resolved_python="$(command -v "$DRAFT_KV_PYTHON" 2>/dev/null || true)"
  [[ -n "$resolved_python" ]] || die "missing Draft-KV Python on PATH: $DRAFT_KV_PYTHON"
  DRAFT_KV_PYTHON="$resolved_python"
fi
[[ -x "$DRAFT_KV_PYTHON" ]] || die "Draft-KV Python is not executable: $DRAFT_KV_PYTHON"
export DRAFT_KV_PYTHON
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "invalid GPU id: $GPU_ID"
[[ -d "$SHARER" ]] || die "missing Sharer model: $SHARER"
[[ -d "$RECEIVER" ]] || die "missing Receiver model: $RECEIVER"
[[ -f "$CHECKPOINT" ]] || die "missing Stage 3 checkpoint: $CHECKPOINT"
[[ -n "$EVAL_CONDITIONS" ]] || die "DRAFT_KV_EVAL_CONDITIONS must not be empty"
IFS=$' \t,' read -r -a CONDITION_ARGS <<<"$EVAL_CONDITIONS"
[[ ${#CONDITION_ARGS[@]} -ge 1 ]] || die "DRAFT_KV_EVAL_CONDITIONS must not be empty"
for dataset in "${DATASET_LIST[@]}"; do
  file="$(adapted_file "$dataset")"
  [[ -f "$file" ]] || die "dataset not prepared: $dataset (expected $file)"
done

mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DRAFT_KV_ROOT
export MATCHED_OUTPUT_ROOT
export STANDALONE_OUTPUT_ROOT
export CHECKPOINT
export SHARER
export RECEIVER
export LAYER_MAPPING
export DATASETS
export SMOKE
export EVAL_CONDITIONS

cd "$REPO_ROOT"

extra_args=()
if [[ "$SMOKE" == "1" ]]; then
  extra_args+=(--max-examples 20)
  echo "SMOKE=1: limiting each benchmark to 20 examples."
fi

for dataset in "${DATASET_LIST[@]}"; do
  limit="$(input_limit "$dataset")"
  dlimit="$(draft_limit "$dataset")"
  dbatch="$(draft_batch "$dataset")"
  ebatch="$(eval_batch "$dataset")"
  log_file="$LOG_ROOT/matched_stage3_${dataset}.log"

  echo
  echo "[matched] Stage 3 on $dataset"
  echo "  input-limit=$limit draft-cap=$dlimit draft-batch=$dbatch eval-batch=$ebatch"
  echo "  log=$log_file"

  status=0
  "$DRAFT_KV_PYTHON" script/draft_kv/run_stage3_eval.py \
    --device cuda:0 \
    --dataset "$dataset" \
    --data-root "$DATA_ROOT" \
    --model-pair custom \
    --sharer "$SHARER" \
    --receiver "$RECEIVER" \
    --layer-mapping "$LAYER_MAPPING" \
    --checkpoint "$CHECKPOINT" \
    --output-root "$MATCHED_OUTPUT_ROOT" \
    --eval-batch-size "$ebatch" \
    --max-sharer-input-tokens "$limit" \
    --max-receiver-input-tokens "$limit" \
    --draft-max-new-tokens "$dlimit" \
    --draft-batch-size "$dbatch" \
    --conditions "${CONDITION_ARGS[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file" || status=$?

  if [[ $status -ne 0 && "$SMOKE" != "1" && $ebatch -gt 1 ]]; then
    half=$(( ebatch / 2 ))
    echo "[retry] $dataset failed with status $status; retrying eval-batch=$half" >&2
    "$DRAFT_KV_PYTHON" script/draft_kv/run_stage3_eval.py \
      --device cuda:0 \
      --dataset "$dataset" \
      --data-root "$DATA_ROOT" \
      --model-pair custom \
      --sharer "$SHARER" \
      --receiver "$RECEIVER" \
      --layer-mapping "$LAYER_MAPPING" \
      --checkpoint "$CHECKPOINT" \
      --output-root "$MATCHED_OUTPUT_ROOT" \
      --eval-batch-size "$half" \
      --max-sharer-input-tokens "$limit" \
      --max-receiver-input-tokens "$limit" \
      --draft-max-new-tokens "$dlimit" \
      --draft-batch-size "$dbatch" \
      --conditions "${CONDITION_ARGS[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee -a "$log_file" || \
        die "$dataset failed at eval-batch $ebatch and $half"
  elif [[ $status -ne 0 ]]; then
    die "$dataset failed; inspect $log_file"
  fi
  echo "[matched] $dataset completed"
done

if [[ "$SKIP_STANDALONE" != "1" ]]; then
  standalone_extra=()
  if [[ "$SMOKE" == "1" ]]; then
    standalone_extra+=(--max-examples 20)
  fi
  standalone_log="$LOG_ROOT/standalone_$(basename "$SHARER").log"
  echo
  echo "[standalone] $(basename "$SHARER") on $DATASETS; log=$standalone_log"
  "$DRAFT_KV_PYTHON" script/draft_kv/eval_standalone_sharer.py \
    --model "$SHARER" \
    --datasets "$DATASETS" \
    --data-root "$DATA_ROOT" \
    --output-root "$STANDALONE_OUTPUT_ROOT" \
    --eval-batch-size "${DRAFT_KV_STANDALONE_EVAL_BATCH_SIZE:-48}" \
    --max-input-tokens 1536 \
    "${standalone_extra[@]}" \
    --device cuda:0 \
    2>&1 | tee "$standalone_log"
fi

SUMMARY_PATH="$LOG_ROOT/summary.txt"
"$DRAFT_KV_PYTHON" - <<'PY' | tee "$SUMMARY_PATH"
import glob
import hashlib
import json
import os

root = os.environ["DRAFT_KV_ROOT"]
matched_root = os.environ["MATCHED_OUTPUT_ROOT"]
standalone_root = os.environ["STANDALONE_OUTPUT_ROOT"]
checkpoint = os.path.realpath(os.environ["CHECKPOINT"])
digest = hashlib.sha256()
with open(checkpoint, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
checkpoint_sha = digest.hexdigest()
datasets = [value for value in os.environ["DATASETS"].split(",") if value]
sharer_name = os.path.basename(os.path.realpath(os.environ["SHARER"]))
receiver_name = os.path.basename(os.path.realpath(os.environ["RECEIVER"]))
conditions = [
    value.strip().capitalize()
    for value in os.environ["EVAL_CONDITIONS"].replace(",", " ").split()
    if value.strip()
]

columns = f"{'dataset':<12}{'Sharer':>9}"
for condition in conditions:
    columns += f"{condition:>11}"
if "Matched" in conditions and "Deranged" in conditions:
    if "Zero" in conditions:
        columns += f"{'M-Z':>9}{'M-D':>9}{'D-Z':>9}"
    else:
        columns += f"{'M-D':>9}"
columns += f"{'n':>7}"
header = columns
print(f"=== Stage 3: {sharer_name} -> {receiver_name} ===")
print(header)
print("-" * len(header))

for dataset in datasets:
    pattern = (
        f"{matched_root}/evaluations/{dataset}/"
        f"{sharer_name}_to_{receiver_name}_*/*/*/eval_result.json"
    )
    candidates = []
    for path in glob.glob(pattern):
        try:
            result = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if (
            os.path.realpath(str(result.get("checkpoint", ""))) == checkpoint
            and result.get("checkpoint_sha256") == checkpoint_sha
            and (os.environ["SMOKE"] == "1" or int(result.get("count", 0)) > 20)
        ):
            candidates.append((os.path.getmtime(path), path, result))
    if not candidates:
        raise SystemExit(f"{dataset}: no matching result for the trained checkpoint")
    _, path, result = max(candidates)
    accuracy = result["accuracy"]

    standalone = ""
    standalone_pattern = (
        f"{standalone_root}/models/{sharer_name}/*/{dataset}/*/eval_result.json"
    )
    standalone_files = sorted(
        glob.glob(standalone_pattern), key=os.path.getmtime
    )
    if standalone_files:
        standalone_result = json.load(
            open(standalone_files[-1], encoding="utf-8")
        )
        standalone = f"{float(standalone_result['accuracy']) * 100:9.2f}"
    else:
        standalone = f"{'--':>9}"

    def percent(condition):
        if condition not in accuracy:
            return f"{'--':>11}"
        return f"{float(accuracy[condition]) * 100:11.2f}"

    def gap(left, right):
        if left not in accuracy or right not in accuracy:
            return f"{'--':>9}"
        return (
            f"{(float(accuracy[left]) - float(accuracy[right])) * 100:+9.2f}"
        )

    count = int(result["count"])
    row = f"{dataset:<12}{standalone}"
    for condition in conditions:
        row += percent(condition)
    if "Matched" in conditions and "Deranged" in conditions:
        if "Zero" in conditions:
            row += (
                gap("Matched", "Zero")
                + gap("Matched", "Deranged")
                + gap("Deranged", "Zero")
            )
        else:
            row += gap("Matched", "Deranged")
    row += f"{count:7d}"
    print(row)
    print(f"  result: {path}")
PY

echo
echo "All Stage 3 MC evaluations completed."
echo "  checkpoint: $CHECKPOINT"
echo "  logs:       $LOG_ROOT"
echo "  summary:    $SUMMARY_PATH"
echo "  outputs:    $MATCHED_OUTPUT_ROOT/evaluations"
