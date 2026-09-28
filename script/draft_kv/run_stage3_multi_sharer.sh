#!/usr/bin/env bash
# Train and evaluate Stage 3 (Matched CE + one-sided Deranged
# protection) for additional Sharer -> Qwen2.5-0.5B-Instruct pairs.
#
# Every pair continues from its completed OpenHermes Stage-2 last.pt and
# reuses the ARC MC bundle and reconstruction replay bundle produced by the
# matching three-stage run:
#   <three_stage_root>/stage2_openhermes/train/last.pt   initial checkpoint
#   <three_stage_root>/stage1_reconstruction/data        replay bundle
#   <three_stage_root>/stage3_mc/data                    ARC MC bundle
#
# Usage:
#   bash script/draft_kv/run_stage3_multi_sharer.sh \
#     --sharers qwen3-1.7b,qwen3-4b,qwen3-8b,llama3.2-3b-instruct \
#     --gpu-id 1
#
# Evaluation covers Matched and Deranged only by default (DRAFT_KV_EVAL_CONDITIONS
# overrides this); Zero, Sharer-only, Draft-only and Text2Text values are
# reused from the master results table.  Completed training runs are reused,
# interrupted ones are not resumed (the Stage 3 trainer keeps no optimizer
# resume state).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DRAFT_KV_DATA_ROOT="${DRAFT_KV_DATA_ROOT:-/workspace/draft-kv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${DRAFT_KV_GPU_ID:-1}"
SHARERS=""
# The private-workspace tree is quota-limited and can be full; training and
# evaluation outputs therefore default to the local writable run root used by
# the Stage-3 Stage 3 workflows.  Dataset, model and Sharer-draft caches are still
# read from the shared tree.
RUN_ROOT="${DRAFT_KV_STAGE3_RUN_ROOT:-/workspace/draft-kv-runs}"
MATCHED_OUTPUT_ROOT="${DRAFT_KV_MATCHED_OUTPUT_ROOT:-}"
SHARER_CACHE_SOURCE="${DRAFT_KV_SHARER_CACHE_SOURCE:-$DRAFT_KV_DATA_ROOT/downstream_mc/sharer_cache}"
EVAL=1
DRY_RUN=0
EXTRA_TRAIN_ARGS=()
MAX_MC_UPDATES=""
PROTECTION_WEIGHT=""
PROTECTION_TOLERANCE=""
DATASETS="${DRAFT_KV_DATASETS:-openbookqa,arc-c,ceval,arc-e,mmlu-redux}"
EVAL_CONDITIONS="${DRAFT_KV_EVAL_CONDITIONS:-matched deranged}"

usage() {
  cat <<'EOF'
Usage:
  bash script/draft_kv/run_stage3_multi_sharer.sh --sharers LIST [options]

Options:
  --sharers LIST             Comma-separated Sharers, one of:
                               qwen3-1.7b
                               qwen3-4b
                               qwen3-8b
                               llama3.2-3b-instruct
  --gpu-id N                 Physical GPU exposed as cuda:0 (default: 1)
  --run-root PATH            Directory receiving stage3_mc_<tag> runs
                             (default: /workspace/draft-kv-runs)
  --datasets LIST            Comma-separated downstream datasets (default:
                             openbookqa,arc-c,ceval,arc-e,mmlu-redux)
  --eval-conditions LIST     Receiver conditions for the downstream pass
                             (default: "matched deranged")
  --max-mc-updates N         Override the 4000-update MC schedule
  --protection-weight W      Override the 0.1 Deranged protection weight
  --protection-tolerance T   Override the 0.1 nats hinge tolerance
  --trainer-arg ARG          Extra argument forwarded to the Stage 3 trainer
                             (repeatable)
  --skip-eval                Train only; leave evaluation to a later call
  --python PATH              Python executable (or set PYTHON_BIN)
  --dry-run                  Print resolved commands and exit
  -h, --help                 Show this help
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sharers) SHARERS="${2:?Missing value for --sharers}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:?Missing value for --gpu-id}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?Missing value for --run-root}"; shift 2 ;;
    --datasets) DATASETS="${2:?Missing value for --datasets}"; shift 2 ;;
    --eval-conditions) EVAL_CONDITIONS="${2:?Missing value for --eval-conditions}"; shift 2 ;;
    --max-mc-updates) MAX_MC_UPDATES="${2:?Missing value for --max-mc-updates}"; shift 2 ;;
    --protection-weight) PROTECTION_WEIGHT="${2:?Missing value for --protection-weight}"; shift 2 ;;
    --protection-tolerance) PROTECTION_TOLERANCE="${2:?Missing value for --protection-tolerance}"; shift 2 ;;
    --trainer-arg) EXTRA_TRAIN_ARGS+=("${2:?Missing value for --trainer-arg}"); shift 2 ;;
    --skip-eval) EVAL=0; shift ;;
    --python) PYTHON_BIN="${2:?Missing value for --python}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$SHARERS" ]] || die "--sharers is required (use --help)"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "invalid --gpu-id: $GPU_ID"
# PYTHON_BIN may be an executable path or a bare command name on PATH.
if [[ "$PYTHON_BIN" != */* ]]; then
  resolved_python="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
  [[ -n "$resolved_python" ]] || die "missing python on PATH: $PYTHON_BIN"
  PYTHON_BIN="$resolved_python"
fi
[[ -x "$PYTHON_BIN" ]] || die "python is not executable: $PYTHON_BIN"
RUN_ROOT="$(realpath -m "$RUN_ROOT")"
[[ -n "$MATCHED_OUTPUT_ROOT" ]] || MATCHED_OUTPUT_ROOT="$RUN_ROOT/downstream_mc"
MATCHED_OUTPUT_ROOT="$(realpath -m "$MATCHED_OUTPUT_ROOT")"
mkdir -p "$RUN_ROOT" "$MATCHED_OUTPUT_ROOT"

# Sharer draft caches are receiver-agnostic and independent of the Stage-3
# checkpoint.  Import the completed caches from the shared tree once so the
# new checkpoints are scored without regenerating drafts; the evaluator then
# rewrites the same rows into its local cache directory.
if [[ -n "$SHARER_CACHE_SOURCE" && -d "$SHARER_CACHE_SOURCE" ]]; then
  IFS=',' read -r -a cache_dataset_list <<< "$DATASETS"
  for cache_dataset in "${cache_dataset_list[@]}"; do
    cache_dataset="${cache_dataset// /}"
    [[ -n "$cache_dataset" ]] || continue
    if [[ -d "$SHARER_CACHE_SOURCE/$cache_dataset" && \
          ! -e "$MATCHED_OUTPUT_ROOT/sharer_cache/$cache_dataset" ]]; then
      mkdir -p "$MATCHED_OUTPUT_ROOT/sharer_cache"
      echo "[Stage 3] importing shared Sharer draft caches for $cache_dataset"
      cp -a "$SHARER_CACHE_SOURCE/$cache_dataset" \
        "$MATCHED_OUTPUT_ROOT/sharer_cache/$cache_dataset"
    fi
  done
  unset cache_dataset_list
fi

[[ -f "$REPO_ROOT/script/draft_kv/run_stage3_training.sh" ]] || \
  die "missing Stage 3 training entry point"
[[ -f "$REPO_ROOT/script/draft_kv/run_stage3_eval_all.sh" ]] || \
  die "missing Stage 3 evaluation entry point"

IFS=',' read -r -a sharer_list <<< "$SHARERS"
FAILED=()

for sharer in "${sharer_list[@]}"; do
  case "$sharer" in
    qwen3-1.7b)
      THREE_STAGE="three_stage_qwen3_17b_sharer_to_qwen25_05b_receiver"
      TAG="qwen3_17b"; MICROBATCH=4; GRAD_ACCUM=4
      MMLU_DRAFT_BATCH=32; DRAFT_BATCH=32
      ;;
    qwen3-4b)
      THREE_STAGE="three_stage_qwen3_4b_sharer_to_qwen25_05b_receiver_6000"
      TAG="qwen3_4b"; MICROBATCH=4; GRAD_ACCUM=4
      MMLU_DRAFT_BATCH=32; DRAFT_BATCH=32
      ;;
    qwen3-8b)
      THREE_STAGE="three_stage_qwen3_8b_sharer_to_qwen25_05b_receiver"
      TAG="qwen3_8b"; MICROBATCH=2; GRAD_ACCUM=8
      MMLU_DRAFT_BATCH=16; DRAFT_BATCH=32
      ;;
    llama3.2-3b-instruct)
      THREE_STAGE="three_stage_llama32_3b_sharer_to_qwen25_05b_receiver"
      TAG="llama32_3b"; MICROBATCH=4; GRAD_ACCUM=4
      MMLU_DRAFT_BATCH=16; DRAFT_BATCH=32
      ;;
    *)
      die "unsupported Sharer: $sharer"
      ;;
  esac

  BASE="$DRAFT_KV_DATA_ROOT/$THREE_STAGE"
  S2_LAST="$BASE/stage2_openhermes/train/last.pt"
  S1_DATA="$BASE/stage1_reconstruction/data"
  S3_DATA="$BASE/stage3_mc/data"
  ROOT="$RUN_ROOT/stage3_mc_training_$TAG"
  TRAIN_DIR="$ROOT/train"
  EVAL_LOG_ROOT="$ROOT/eval_logs"
  RESULT="$TRAIN_DIR/train_result.json"

  for required in \
      "$S2_LAST" \
      "$S1_DATA/data_manifest.json" \
      "$S1_DATA/reconstruction_records.jsonl" \
      "$S3_DATA/data_manifest.json" \
      "$S3_DATA/prepare_result.json"; do
    [[ -f "$required" ]] || die "missing required artifact: $required"
  done

  # The MC bundle registered the Sharer, Receiver and Receiver:Sharer mapping
  # that the trainer will load; reuse them for the downstream evaluation.
  read -r SHARER_PATH RECEIVER_PATH LAYER_MAPPING < <(
    "$PYTHON_BIN" - "$S3_DATA/data_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
mapping = ",".join(
    f"{target}:{source}"
    for target, source in sorted(
        ((int(key), int(value)) for key, value in manifest["layer_mapping"].items())
    )
)
print(manifest["sharer"], manifest["receiver"], mapping)
PY
  )
  [[ -d "$SHARER_PATH" ]] || die "missing Sharer model: $SHARER_PATH"
  [[ -d "$RECEIVER_PATH" ]] || die "missing Receiver model: $RECEIVER_PATH"

  train_cmd=(
    bash "$REPO_ROOT/script/draft_kv/run_stage3_training.sh"
    --gpu-id "$GPU_ID"
    --microbatch "$MICROBATCH"
    --grad-accum "$GRAD_ACCUM"
  )
  [[ -z "$MAX_MC_UPDATES" ]] || train_cmd+=(--max-mc-updates "$MAX_MC_UPDATES")
  [[ -z "$PROTECTION_WEIGHT" ]] || train_cmd+=(--protection-weight "$PROTECTION_WEIGHT")
  [[ -z "$PROTECTION_TOLERANCE" ]] || train_cmd+=(--protection-tolerance "$PROTECTION_TOLERANCE")
  [[ ${#EXTRA_TRAIN_ARGS[@]} -eq 0 ]] || train_cmd+=("${EXTRA_TRAIN_ARGS[@]}")

  echo "[Stage 3] sharer=$sharer gpu=$GPU_ID run=$ROOT"
  echo "[Stage 3]   stage2=$S2_LAST"
  echo "[Stage 3]   microbatch=$MICROBATCH grad_accum=$GRAD_ACCUM" \
       "draft_batch=$DRAFT_BATCH mmlu_draft_batch=$MMLU_DRAFT_BATCH"
  if (( DRY_RUN )); then
    printf '[Stage 3]   '
    printf '%q ' env \
      HF_HOME="$RUN_ROOT/hf_home" \
      DRAFT_KV_STAGE3_DATA_DIR="$S3_DATA" \
      DRAFT_KV_STAGE3_INITIAL_STAGE2="$S2_LAST" \
      DRAFT_KV_STAGE3_RECON_DATA="$S1_DATA" \
      DRAFT_KV_STAGE3_OUTPUT_DIR="$TRAIN_DIR" \
      "${train_cmd[@]}"
    printf '\n'
    if (( EVAL )); then
      printf '[Stage 3]   '
      printf '%q ' env \
        HF_HOME="$RUN_ROOT/hf_home" \
        DRAFT_KV_GPU_ID="$GPU_ID" \
        DRAFT_KV_CHECKPOINT="$TRAIN_DIR/best.pt" \
        DRAFT_KV_SHARER="$SHARER_PATH" \
        DRAFT_KV_RECEIVER="$RECEIVER_PATH" \
        DRAFT_KV_LAYER_MAPPING="$LAYER_MAPPING" \
        DRAFT_KV_MATCHED_OUTPUT_ROOT="$MATCHED_OUTPUT_ROOT" \
        DRAFT_KV_DATASETS="$DATASETS" \
        DRAFT_KV_EVAL_CONDITIONS="$EVAL_CONDITIONS" \
        DRAFT_KV_SKIP_STANDALONE=1 \
        DRAFT_KV_DRAFT_BATCH_SIZE_MMLU_REDUX="$MMLU_DRAFT_BATCH" \
        DRAFT_KV_DRAFT_BATCH_SIZE="$DRAFT_BATCH" \
        DRAFT_KV_STAGE3_EVAL_LOG_ROOT="$EVAL_LOG_ROOT" \
        bash "$REPO_ROOT/script/draft_kv/run_stage3_eval_all.sh"
      printf '\n'
    fi
    continue
  fi

  status=0
  if [[ -f "$RESULT" && -f "$TRAIN_DIR/best.pt" ]]; then
    echo "[Stage 3] reusing completed training run: $TRAIN_DIR"
  else
    if [[ -e "$TRAIN_DIR/last.pt" ]]; then
      die "training run is incomplete and Stage 3 has no optimizer resume: $TRAIN_DIR"
    fi
    if DRAFT_KV_STAGE3_DATA_DIR="$S3_DATA" \
       DRAFT_KV_STAGE3_INITIAL_STAGE2="$S2_LAST" \
       DRAFT_KV_STAGE3_RECON_DATA="$S1_DATA" \
       DRAFT_KV_STAGE3_OUTPUT_DIR="$TRAIN_DIR" \
       HF_HOME="$RUN_ROOT/hf_home" \
       "${train_cmd[@]}"; then
      echo "[Stage 3] training finished for $sharer"
    else
      status=$?
      echo "[Stage 3] FAILED training $sharer (status=$status); inspect $TRAIN_DIR/logs" >&2
      FAILED+=("$sharer:train")
      continue
    fi
  fi

  if (( EVAL )); then
    if DRAFT_KV_CHECKPOINT="$TRAIN_DIR/best.pt" \
       DRAFT_KV_GPU_ID="$GPU_ID" \
       DRAFT_KV_SHARER="$SHARER_PATH" \
       DRAFT_KV_RECEIVER="$RECEIVER_PATH" \
       DRAFT_KV_LAYER_MAPPING="$LAYER_MAPPING" \
       DRAFT_KV_MATCHED_OUTPUT_ROOT="$MATCHED_OUTPUT_ROOT" \
       DRAFT_KV_DATASETS="$DATASETS" \
       DRAFT_KV_EVAL_CONDITIONS="$EVAL_CONDITIONS" \
       DRAFT_KV_SKIP_STANDALONE=1 \
       DRAFT_KV_DRAFT_BATCH_SIZE_MMLU_REDUX="$MMLU_DRAFT_BATCH" \
       DRAFT_KV_DRAFT_BATCH_SIZE="$DRAFT_BATCH" \
       DRAFT_KV_STAGE3_EVAL_LOG_ROOT="$EVAL_LOG_ROOT" \
       HF_HOME="$RUN_ROOT/hf_home" \
       bash "$REPO_ROOT/script/draft_kv/run_stage3_eval_all.sh"; then
      echo "[Stage 3] evaluation finished for $sharer"
    else
      status=$?
      echo "[Stage 3] FAILED evaluation $sharer (status=$status); inspect $EVAL_LOG_ROOT" >&2
      FAILED+=("$sharer:eval")
    fi
  fi
done

if (( DRY_RUN )); then
  echo "[Stage 3] dry run complete"
  exit 0
fi

if ((${#FAILED[@]})); then
  echo "[Stage 3] failed stages: ${FAILED[*]}" >&2
  exit 1
fi

echo "[Stage 3] all requested Sharers finished"
