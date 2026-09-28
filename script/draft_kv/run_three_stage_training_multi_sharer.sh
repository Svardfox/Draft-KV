#!/usr/bin/env bash
# Run the complete three-stage Draft-KV training pipeline with a
# configurable Receiver (default Qwen2.5-0.5B-Instruct) and one of the
# supported Sharers.
#
# Stage 1: OpenHermes textual reconstruction (overfit gate + long pilot)
# Stage 2: OpenHermes answer training with reconstruction replay
# Stage 3: ARC-E/ARC-C multiple-choice option training with replay
#
# The wrapper deliberately keeps every stage in a separate directory.  A
# completed stage is reused on rerun; all underlying programs still authenticate
# manifests, model paths, layer mappings and checkpoint lineage.

set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="${DRAFT_KV_TRAINING_ORIGINAL_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${DRAFT_KV_PYTHON:-python3}"

# Mapping syntax is Receiver-layer:Sharer-layer. The 14:18,16:20,18:22,20:24
# default is the proven configuration for the 24-layer 0.5B Receiver; each
# Receiver case below supplies its own depth-proportional default, and an
# explicit --layer-mapping always wins.
RECEIVER_MODEL="qwen2.5-0.5b-instruct"
RECEIVER=""
RUN_BASE=""
SHARER_MODEL=""
SHARER=""
LAYER_MAPPING="14:18,16:20,18:22,20:24"
LAYER_MAPPING_EXPLICIT=0
TRAIN_MICROBATCH_SCALE=1
OPENHERMES="/workspace/datasets/OpenHermes-2.5-500k/openhermes2_5_500k.json"
DATA_ROOT="/workspace/datasets"
OUTPUT_ROOT=""
GPU_ID="0"
ALLOW_STAGE1_GATE_NO_GO=0
S1_OVERFIT_MICROBATCH=1
S1_PILOT_MICROBATCH=1
S1_PILOT_EVAL_BATCH=1
S2_MICROBATCH=4
S3_MICROBATCH=4
S2_DRAFT_BATCH=1
S3_DRAFT_BATCH=16
S1_OVERFIT_GRAD_ACCUM=32
S1_PILOT_GRAD_ACCUM=64
S1_PILOT_UPDATES=6000
S1_PILOT_UPDATES_OVERRIDE=""
S1_CHECKPOINT_SELECTION="best"
S2_DRAFT_BATCH_OVERRIDE=""
S3_DRAFT_BATCH_OVERRIDE=""
S2_GRAD_ACCUM=8
S3_GRAD_ACCUM=4

usage() {
  cat <<'EOF'
Usage:
  bash script/draft_kv/run_three_stage_training_multi_sharer.sh \
    --sharer-model MODEL [options]

Runs Stage 1 reconstruction, Stage 2 OpenHermes answer training, and Stage 3
ARC option training in one pipeline. The Receiver defaults to
Qwen2.5-0.5B-Instruct (24 layers) and can be switched with --receiver-model.

  Required:
  --sharer-model MODEL         One of:
                                 qwen3-8b
                                 qwen3-4b
                                 qwen3-1.7b
                                 qwen2.5-coder-1.5b-instruct
                                 qwen2.5-math-1.5b-instruct
                                 llama3.2-3b-instruct

Options:
  --receiver-model MODEL       Receiver (default: qwen2.5-0.5b-instruct):
                                 qwen2.5-0.5b-instruct (24 layers)
                                 qwen3-4b (36 layers; depth-proportional
                                   default layer mapping and halved training
                                   microbatches are applied automatically)
  --layer-mapping MAP          Receiver:Sharer pairs (default depends on the
                               Receiver: 14:18,16:20,18:22,20:24 for the
                               24-layer 0.5B; 21:23,24:26,27:29,30:32 for the
                               36-layer Qwen3-4B)
  --openhermes PATH            OpenHermes JSON/JSONL source
  --data-root PATH             Root containing ai2_arc/ for Stage 3
  --output-root PATH           Root for all data, checkpoints and logs
  --gpu-id ID                  Physical GPU exposed as cuda:0 (default: 0)
  --stage1-pilot-updates N     Stage-1 Pilot limit (4000, 6000, or 16000); use
                               16000 when resuming the original long-run artifact
  --stage1-checkpoint-selection NAME
                               Stage-1 checkpoint used to initialize Stage 2:
                               best (default) or last. Use last with a 4000-update
                               Pilot to run the update-4000 ablation.
  --stage3-draft-batch-size N  Sharer generation batch for Stage 3 (default 16;
                               1.5B/1.7B Sharers use 32)
  --stage2-draft-batch-size N  Sharer generation batch for Stage 2 (profile
                               default; 4B/8B use 16, 1.5B/1.7B use 32)
  --train-microbatch-scale N   Divide every training microbatch by N and
                               multiply grad-accum by N (effective batches
                               unchanged; default 1). Use 2 or 4 only when a
                               training step OOMs. The qwen3-4b Receiver
                               already applies an internal factor of 2.
  --python PATH                Python executable (or set DRAFT_KV_PYTHON)
  --allow-stage1-gate-no-go   Continue after Stage-1 pilot Gate-val NO_GO
  -h, --help                  Show this help

The flag above does not bypass the mandatory Stage-1 Overfit GO prerequisite;
the Stage-1 trainer itself refuses to start the pilot without that result.
Stage-2 and Stage-3 are exploratory continuation stages, so Gate-val is logged
and enforced only when --allow-stage1-gate-no-go is omitted.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_value() {
  [[ $# -ge 2 ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sharer-model)
      need_value "$@"
      SHARER_MODEL="$2"
      shift 2
      ;;
    --receiver-model)
      need_value "$@"
      RECEIVER_MODEL="$2"
      shift 2
      ;;
    --layer-mapping)
      need_value "$@"
      LAYER_MAPPING="$2"
      LAYER_MAPPING_EXPLICIT=1
      shift 2
      ;;
    --openhermes)
      need_value "$@"
      OPENHERMES="$2"
      shift 2
      ;;
    --data-root)
      need_value "$@"
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-root)
      need_value "$@"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --gpu-id)
      need_value "$@"
      GPU_ID="$2"
      shift 2
      ;;
    --stage1-pilot-updates)
      need_value "$@"
      S1_PILOT_UPDATES_OVERRIDE="$2"
      shift 2
      ;;
    --stage1-checkpoint-selection)
      need_value "$@"
      S1_CHECKPOINT_SELECTION="$2"
      shift 2
      ;;
    --stage3-draft-batch-size)
      need_value "$@"
      S3_DRAFT_BATCH_OVERRIDE="$2"
      shift 2
      ;;
    --stage2-draft-batch-size)
      need_value "$@"
      S2_DRAFT_BATCH_OVERRIDE="$2"
      shift 2
      ;;
    --train-microbatch-scale)
      need_value "$@"
      TRAIN_MICROBATCH_SCALE="$2"
      shift 2
      ;;
    --python)
      need_value "$@"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --allow-stage1-gate-no-go)
      ALLOW_STAGE1_GATE_NO_GO=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

case "$SHARER_MODEL" in
  qwen3-8b)
    SHARER="/workspace/models/Qwen3-8B"
    RUN_BASE="qwen3_8b_sharer"
    # Preserve the established effective optimizer batches (32/64/32/16)
    # while keeping the frozen 8B Sharer comfortably within one 96GB GPU.
    S1_OVERFIT_MICROBATCH=4
    S1_OVERFIT_GRAD_ACCUM=8
    S1_PILOT_MICROBATCH=4
    S1_PILOT_GRAD_ACCUM=16
    S1_PILOT_EVAL_BATCH=1
    S2_MICROBATCH=4
    S2_GRAD_ACCUM=8
    S3_MICROBATCH=4
    S3_GRAD_ACCUM=4
    S2_DRAFT_BATCH=16
    S3_DRAFT_BATCH=16
    ;;
  qwen3-4b)
    SHARER="/workspace/models/Qwen3-4B"
    RUN_BASE="qwen3_4b_sharer"
    S1_OVERFIT_MICROBATCH=8
    S1_OVERFIT_GRAD_ACCUM=4
    S1_PILOT_MICROBATCH=8
    S1_PILOT_GRAD_ACCUM=8
    S2_DRAFT_BATCH=16
    S3_DRAFT_BATCH=16
    ;;
  qwen3-1.7b)
    SHARER="/workspace/models/Qwen3-1.7B"
    RUN_BASE="qwen3_17b_sharer"
    # Same 28-layer Qwen3 family as the 4B profile; the smaller Sharer leaves
    # more memory headroom, so it uses the 1.5B profile's draft batches.
    S1_OVERFIT_MICROBATCH=8
    S1_OVERFIT_GRAD_ACCUM=4
    S1_PILOT_MICROBATCH=8
    S1_PILOT_GRAD_ACCUM=8
    S2_DRAFT_BATCH=32
    S3_DRAFT_BATCH=32
    ;;
  qwen2.5-coder-1.5b-instruct)
    SHARER="/workspace/models/Qwen2.5-Coder-1.5B-Instruct"
    RUN_BASE="qwen25_coder_15b_sharer"
    S1_OVERFIT_MICROBATCH=8
    S1_PILOT_MICROBATCH=32
    S1_PILOT_EVAL_BATCH=8
    S2_MICROBATCH=8
    S3_MICROBATCH=8
    S2_DRAFT_BATCH=32
    S3_DRAFT_BATCH=32
    S1_OVERFIT_GRAD_ACCUM=4
    S1_PILOT_GRAD_ACCUM=2
    S1_PILOT_UPDATES=6000
    S2_GRAD_ACCUM=4
    S3_GRAD_ACCUM=2
    ;;
  qwen2.5-math-1.5b-instruct)
    SHARER="/workspace/models/Qwen2.5-Math-1.5B-Instruct"
    RUN_BASE="qwen25_math_15b_sharer"
    S1_OVERFIT_MICROBATCH=4
    S1_PILOT_MICROBATCH=32
    S1_PILOT_EVAL_BATCH=8
    S2_MICROBATCH=4
    S3_MICROBATCH=4
    S2_DRAFT_BATCH=32
    S3_DRAFT_BATCH=32
    S1_OVERFIT_GRAD_ACCUM=8
    S1_PILOT_GRAD_ACCUM=2
    S1_PILOT_UPDATES=6000
    S2_GRAD_ACCUM=8
    S3_GRAD_ACCUM=4
    ;;
  llama3.2-3b-instruct)
    SHARER="/workspace/models/Llama-3.2-3B-Instruct"
    RUN_BASE="llama32_3b_sharer"
    # Llama-3.2-3B is a 28-layer model with the same KV width (8 heads x 128)
    # as every other Sharer, but hidden_size 3072 makes it the largest 28-layer
    # backbone.  Memory profile follows the 4B/1.7B tier (28-layer batch
    # shapes) while draft generation uses the conservative 4B/8B batch of 16.
    S1_OVERFIT_MICROBATCH=8
    S1_OVERFIT_GRAD_ACCUM=4
    S1_PILOT_MICROBATCH=8
    S1_PILOT_GRAD_ACCUM=8
    S2_DRAFT_BATCH=16
    S3_DRAFT_BATCH=16
    ;;
  "")
    die "--sharer-model is required (use --help)"
    ;;
  *)
    die "unsupported --sharer-model: $SHARER_MODEL (use --help)"
    ;;
esac

# Divide every training microbatch by SCALE and multiply grad-accum by SCALE
# so the effective optimizer batches (32/64/32/16) stay identical. A 4B-class
# Receiver multiplies training activations (36 layers x hidden 2560 x 32 query
# heads) far beyond the 0.5B default; draft generation only loads the Sharer,
# so the draft batches keep their Sharer-tier profile.
scale_train_profile() {
  local scale="${1:-2}"
  [[ "$scale" =~ ^[1-9][0-9]*$ ]] || die "microbatch scale must be a positive integer: $scale"
  if (( scale == 1 )); then
    return 0
  fi
  S1_OVERFIT_MICROBATCH=$(( S1_OVERFIT_MICROBATCH / scale < 1 ? 1 : S1_OVERFIT_MICROBATCH / scale ))
  S1_OVERFIT_GRAD_ACCUM=$(( S1_OVERFIT_GRAD_ACCUM * scale ))
  S1_PILOT_MICROBATCH=$(( S1_PILOT_MICROBATCH / scale < 1 ? 1 : S1_PILOT_MICROBATCH / scale ))
  S1_PILOT_GRAD_ACCUM=$(( S1_PILOT_GRAD_ACCUM * scale ))
  S2_MICROBATCH=$(( S2_MICROBATCH / scale < 1 ? 1 : S2_MICROBATCH / scale ))
  S2_GRAD_ACCUM=$(( S2_GRAD_ACCUM * scale ))
  S3_MICROBATCH=$(( S3_MICROBATCH / scale < 1 ? 1 : S3_MICROBATCH / scale ))
  S3_GRAD_ACCUM=$(( S3_GRAD_ACCUM * scale ))
}

case "$RECEIVER_MODEL" in
  qwen2.5-0.5b-instruct)
    RECEIVER="/workspace/models/Qwen2.5-0.5B-Instruct"
    RECEIVER_TAG="qwen25_05b"
    RECEIVER_DEFAULT_LAYER_MAPPING="14:18,16:20,18:22,20:24"
    RECEIVER_PROFILE_SCALE=1
    ;;
  qwen3-4b)
    RECEIVER="/workspace/models/Qwen3-4B"
    RECEIVER_TAG="qwen3_4b"
    # 36-layer Receiver. The 24-layer defaults (targets 14/16/18/20 = 39-56%
    # depth) sit far too early; scale the proven 0.5B relative depths
    # (58-83%) onto 36 layers (targets 21/24/27/30) and keep Sharer sources
    # at the 0.6B-proven 64-89% band (23/26/29/32) instead of the mid-stack
    # layers that crippled the 4B-Sharer-to-0.5B run.
    RECEIVER_DEFAULT_LAYER_MAPPING="21:23,24:26,27:29,30:32"
    RECEIVER_PROFILE_SCALE=2
    ;;
  "")
    die "--receiver-model is empty (default: qwen2.5-0.5b-instruct; use --help)"
    ;;
  *)
    die "unsupported --receiver-model: $RECEIVER_MODEL (use --help)"
    ;;
esac

if [[ "$LAYER_MAPPING_EXPLICIT" != 1 ]]; then
  LAYER_MAPPING="$RECEIVER_DEFAULT_LAYER_MAPPING"
fi

[[ -n "$RUN_BASE" ]] || die "internal error: RUN_BASE is empty"
RUN_NAME="${RUN_BASE}_to_${RECEIVER_TAG}_receiver"

[[ "$TRAIN_MICROBATCH_SCALE" =~ ^[1-9][0-9]*$ ]] || \
  die "--train-microbatch-scale must be a positive integer"
scale_train_profile "$(( RECEIVER_PROFILE_SCALE * TRAIN_MICROBATCH_SCALE ))"

if [[ -n "$S1_PILOT_UPDATES_OVERRIDE" ]]; then
  case "$S1_PILOT_UPDATES_OVERRIDE" in
    4000|6000|16000)
      S1_PILOT_UPDATES="$S1_PILOT_UPDATES_OVERRIDE"
      ;;
    *)
      die "--stage1-pilot-updates must be 4000, 6000, or 16000"
      ;;
  esac
fi

case "$S1_CHECKPOINT_SELECTION" in
  best|last) ;;
  *) die "--stage1-checkpoint-selection must be best or last" ;;
esac

if [[ -n "$S3_DRAFT_BATCH_OVERRIDE" ]]; then
  [[ "$S3_DRAFT_BATCH_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || \
    die "--stage3-draft-batch-size must be a positive integer"
  S3_DRAFT_BATCH="$S3_DRAFT_BATCH_OVERRIDE"
fi

if [[ -n "$S2_DRAFT_BATCH_OVERRIDE" ]]; then
  [[ "$S2_DRAFT_BATCH_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || \
    die "--stage2-draft-batch-size must be a positive integer"
  S2_DRAFT_BATCH="$S2_DRAFT_BATCH_OVERRIDE"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="/workspace/draft-kv/three_stage_${RUN_NAME}"
fi

# PYTHON_BIN may be an executable path or a bare command name on PATH.
if [[ "$PYTHON_BIN" != */* ]]; then
  resolved_python="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
  [[ -n "$resolved_python" ]] || die "Python executable not found on PATH: $PYTHON_BIN"
  PYTHON_BIN="$resolved_python"
fi
[[ -x "$PYTHON_BIN" ]] || die "Python executable is not executable: $PYTHON_BIN"
[[ -f "$OPENHERMES" ]] || die "OpenHermes source does not exist: $OPENHERMES"
[[ -f "$RECEIVER/config.json" ]] || die "Receiver config is missing: $RECEIVER/config.json"
[[ -f "$SHARER/config.json" ]] || die "Sharer config is missing: $SHARER/config.json"
[[ -f "$DATA_ROOT/ai2_arc/manifest.json" ]] || die "ARC manifest is missing: $DATA_ROOT/ai2_arc/manifest.json"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "GPU id must be a non-negative integer: $GPU_ID"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
mkdir -p "$OUTPUT_ROOT/logs"

# Bash can read a long-running script incrementally. Rewriting the launcher
# while a model subprocess is running can therefore make the parent shell read
# a mixture of old and new text when it resumes. Execute an immutable per-run
# snapshot so later repository edits cannot corrupt an active pipeline.
if [[ "${DRAFT_KV_TRAINING_LAUNCHER_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  LAUNCHER_SNAPSHOT="$OUTPUT_ROOT/logs/launcher_snapshot_$(date +%Y%m%dT%H%M%S)_$$.sh"
  cp -- "${BASH_SOURCE[0]}" "$LAUNCHER_SNAPSHOT"
  chmod 0555 "$LAUNCHER_SNAPSHOT"
  echo "[launcher] executing immutable snapshot: $LAUNCHER_SNAPSHOT"
  export DRAFT_KV_TRAINING_LAUNCHER_SNAPSHOT_ACTIVE=1
  export DRAFT_KV_TRAINING_ORIGINAL_SCRIPT_DIR="$SCRIPT_DIR"
  exec bash "$LAUNCHER_SNAPSHOT" "${ORIGINAL_ARGS[@]}"
fi

# Fail before loading any model if a mapping is outside either model's depth.
# This check also makes the direction of the mapping explicit in the run log.
"$PYTHON_BIN" - "$RECEIVER" "$SHARER" "$LAYER_MAPPING" <<'PY'
import json
import sys
from pathlib import Path

receiver, sharer, mapping_text = sys.argv[1:]

def config(path):
    return json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))

rcfg = config(receiver)
scfg = config(sharer)
try:
    pairs = {}
    for item in mapping_text.split(","):
        left, right = item.strip().split(":", 1)
        target, source = int(left), int(right)
        if target in pairs:
            raise ValueError(f"duplicate Receiver layer {target}")
        pairs[target] = source
except Exception as exc:
    raise SystemExit(f"invalid --layer-mapping {mapping_text!r}: {exc}")
if not pairs:
    raise SystemExit("--layer-mapping cannot be empty")
receiver_depth = int(rcfg["num_hidden_layers"])
sharer_depth = int(scfg["num_hidden_layers"])
bad_targets = [x for x in pairs if x < 0 or x >= receiver_depth]
bad_sources = [x for x in pairs.values() if x < 0 or x >= sharer_depth]
if bad_targets or bad_sources:
    raise SystemExit(
        "layer mapping is outside model depth: "
        f"bad Receiver={bad_targets}, bad Sharer={bad_sources}; "
        f"valid Receiver=0..{receiver_depth - 1}, "
        f"valid Sharer=0..{sharer_depth - 1}"
    )
print(
    "Validated layer mapping (Receiver:Sharer): "
    + ",".join(f"{target}:{source}" for target, source in sorted(pairs.items()))
)
print(
    f"Receiver {rcfg.get('model_type')} depth={receiver_depth}; "
    f"Sharer {scfg.get('model_type')} depth={sharer_depth}"
)
PY

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DEVICE="cuda:0"

S1_ROOT="$OUTPUT_ROOT/stage1_reconstruction"
S1_DATA="$S1_ROOT/data"
S1_OVERFIT_TRAIN="$S1_ROOT/overfit_train"
S1_OVERFIT_EVAL="$S1_ROOT/overfit_eval"
S1_PILOT_TRAIN="$S1_ROOT/pilot_train"
S1_GATE_EVAL="$S1_ROOT/gate_eval"
S2_ROOT="$OUTPUT_ROOT/stage2_openhermes"
S2_CANDIDATE="$S2_ROOT/candidate"
S2_EOS="$S2_ROOT/eos_pool"
S2_TRAIN="$S2_ROOT/train"
S3_ROOT="$OUTPUT_ROOT/stage3_mc"
S3_DATA="$S3_ROOT/data"
S3_TRAIN="$S3_ROOT/train"

run_logged() {
  local log_file="$1"
  shift
  echo
  echo "+ $*"
  "$@" 2>&1 | tee -a "$log_file"
}

complete_files() {
  local root="$1"
  shift
  local name
  for name in "$@"; do
    [[ -f "$root/$name" ]] || return 1
  done
}

json_decision() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("decision", ""))
PY
}

candidate_cache_ready() {
  "$PYTHON_BIN" - "$S2_CANDIDATE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
try:
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    records = root / str(manifest["records_file"])
    drafts = root / str(manifest["draft_cache_file"])
    expected = []
    for split in ("train", "gate_val", "reserve_test"):
        expected.extend(str(value) for value in manifest["split_ids"][split])
    record_ids = []
    for line in records.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record_ids.append(str(json.loads(line)["example_id"]))
    draft_ids = []
    for line in drafts.read_text(encoding="utf-8").splitlines():
        if line.strip():
            draft_ids.append(str(json.loads(line)["example_id"]))
    if record_ids != expected or set(draft_ids) != set(expected):
        raise ValueError("candidate records/cache are incomplete")
except Exception as exc:
    print(f"candidate cache check failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

echo "Three-stage Draft-KV training"
echo "  Choice:   $SHARER_MODEL"
echo "  Sharer:   $SHARER"
echo "  Receiver: $RECEIVER"
echo "  Mapping:  $LAYER_MAPPING"
echo "  Training: Stage1 overfit microbatch/grad_accum $S1_OVERFIT_MICROBATCH/$S1_OVERFIT_GRAD_ACCUM"
echo "            Stage1 pilot microbatch/grad_accum $S1_PILOT_MICROBATCH/$S1_PILOT_GRAD_ACCUM"
echo "            Stage2 microbatch/grad_accum $S2_MICROBATCH/$S2_GRAD_ACCUM"
echo "            Stage3 microbatch/grad_accum $S3_MICROBATCH/$S3_GRAD_ACCUM"
echo "            Stage2/Stage3 draft batch $S2_DRAFT_BATCH/$S3_DRAFT_BATCH"
echo "            Stage1 pilot updates $S1_PILOT_UPDATES"
echo "            Stage2 initialization Stage1 $S1_CHECKPOINT_SELECTION.pt"
echo "  Output:   $OUTPUT_ROOT"
echo "  GPU:      physical $GPU_ID -> $DEVICE"

# Persist the semantic orchestration record. A different pair, mapping, data,
# or training lineage is rejected; physical GPU IDs are operational and may
# change across resumptions, so their history is recorded separately.
"$PYTHON_BIN" - "$OUTPUT_ROOT/run_manifest.json" "$RECEIVER" "$SHARER" "$SHARER_MODEL" "$LAYER_MAPPING" "$OPENHERMES" "$DATA_ROOT" "$GPU_ID" "$S1_PILOT_UPDATES" "$S1_CHECKPOINT_SELECTION" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
current_gpu = str(sys.argv[8])
payload = {
    "protocol": "draft_kv_three_stage_multi_sharer_training",
    "receiver": str(Path(sys.argv[2]).resolve()),
    "sharer": str(Path(sys.argv[3]).resolve()),
    "sharer_model": sys.argv[4],
    "layer_mapping": {
        str(int(item.split(":", 1)[0])): int(item.split(":", 1)[1])
        for item in sys.argv[5].split(",")
    },
    "openhermes": str(Path(sys.argv[6]).resolve()),
    "data_root": str(Path(sys.argv[7]).resolve()),
    "gpu_id": current_gpu,
    "memory_profile": "single_gpu_conservative",
    "stage1_mode": "longrun",
    "stage1_pilot_updates": int(sys.argv[9]),
    "stage2_updates": 4000,
    "stage3_mc_updates": 4000,
}
if sys.argv[10] != "best":
    payload["stage1_checkpoint_selection"] = sys.argv[10]
if path.exists():
    old = json.loads(path.read_text(encoding="utf-8"))

    def semantic(value):
        result = dict(value)
        result.pop("gpu_id", None)
        result.pop("gpu_ids_used", None)
        return result

    if semantic(old) != semantic(payload):
        raise SystemExit(
            f"existing run_manifest.json differs from this command: {path}"
        )
    gpu_ids = {str(value) for value in old.get("gpu_ids_used", [])}
    if old.get("gpu_id") is not None:
        gpu_ids.add(str(old["gpu_id"]))
    gpu_ids.add(current_gpu)
    old["gpu_ids_used"] = sorted(gpu_ids, key=int)
    path.write_text(
        json.dumps(old, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
else:
    payload["gpu_ids_used"] = [current_gpu]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

# ----------------------------- Stage 1 ------------------------------------
if complete_files "$S1_DATA" data_manifest.json reconstruction_records.jsonl prepare_result.json; then
  echo "[Stage 1] reuse completed data preparation"
else
  run_logged "$OUTPUT_ROOT/logs/stage1_prepare.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/prepare_draft_kv_openhermes_reconstruction.py" \
      --data "$OPENHERMES" \
      --receiver "$RECEIVER" \
      --sharer "$SHARER" \
      --layer-mapping "$LAYER_MAPPING" \
      --output-dir "$S1_DATA" \
      --reconstruction-mode longrun \
      --seed 91827 \
      --derangement-seed 71031
fi

if complete_files "$S1_OVERFIT_TRAIN" last.pt best.pt train_history.json train_result.json; then
  echo "[Stage 1] reuse completed Overfit training (microbatch $S1_OVERFIT_MICROBATCH, grad_accum $S1_OVERFIT_GRAD_ACCUM)"
else
  run_logged "$OUTPUT_ROOT/logs/stage1_overfit_train.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/train_draft_kv_openhermes_reconstruction.py" \
      --data-dir "$S1_DATA" \
      --output-dir "$S1_OVERFIT_TRAIN" \
      --mode overfit \
      --max-updates 300 \
      --projector-lr 1e-3 \
      --gate-lr 1e-2 \
      --microbatch "$S1_OVERFIT_MICROBATCH" \
      --grad-accum "$S1_OVERFIT_GRAD_ACCUM" \
      --eval-batch-size 1 \
      --eval-every 50 \
      --log-every 10 \
      --seed 91827 \
      --device "$DEVICE"
fi

if complete_files "$S1_OVERFIT_EVAL" eval_result.json eval_manifest.json per_example_nll.jsonl per_example_generation.jsonl; then
  echo "[Stage 1] reuse completed Overfit evaluation"
else
  run_logged "$OUTPUT_ROOT/logs/stage1_overfit_eval.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/eval_draft_kv_openhermes_reconstruction.py" \
      --data-dir "$S1_DATA" \
      --checkpoint "$S1_OVERFIT_TRAIN/last.pt" \
      --output-dir "$S1_OVERFIT_EVAL" \
      --split overfit \
      --batch-size 1 \
      --generation-examples 32 \
      --seed 91827 \
      --device "$DEVICE"
fi

OVERFIT_DECISION="$(json_decision "$S1_OVERFIT_EVAL/eval_result.json")"
[[ "$OVERFIT_DECISION" == "GO" ]] || die "Stage-1 Overfit gate is $OVERFIT_DECISION; pilot cannot start"
echo "[Stage 1] Overfit gate: GO"

if complete_files "$S1_PILOT_TRAIN" last.pt best.pt train_history.json train_result.json; then
  echo "[Stage 1] reuse completed ${S1_PILOT_UPDATES}-update pilot"
else
  run_logged "$OUTPUT_ROOT/logs/stage1_pilot_train.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/train_draft_kv_openhermes_reconstruction.py" \
      --data-dir "$S1_DATA" \
      --output-dir "$S1_PILOT_TRAIN" \
      --mode pilot \
      --max-updates "$S1_PILOT_UPDATES" \
      --projector-lr 1e-3 \
      --gate-lr 1e-2 \
      --microbatch "$S1_PILOT_MICROBATCH" \
      --grad-accum "$S1_PILOT_GRAD_ACCUM" \
      --eval-batch-size "$S1_PILOT_EVAL_BATCH" \
      --eval-every 1000 \
      --log-every 100 \
      --seed 91827 \
      --require-overfit-go "$S1_OVERFIT_EVAL/eval_result.json" \
      --device "$DEVICE"
fi

if complete_files "$S1_GATE_EVAL" eval_result.json eval_manifest.json per_example_nll.jsonl per_example_generation.jsonl; then
  echo "[Stage 1] reuse completed pilot Gate-val evaluation"
else
  run_logged "$OUTPUT_ROOT/logs/stage1_gate_eval.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/eval_draft_kv_openhermes_reconstruction.py" \
      --data-dir "$S1_DATA" \
      --checkpoint "$S1_PILOT_TRAIN/last.pt" \
      --output-dir "$S1_GATE_EVAL" \
      --split gate_val \
      --batch-size "$S1_PILOT_EVAL_BATCH" \
      --generation-examples 64 \
      --seed 91827 \
      --device "$DEVICE"
fi

GATE_DECISION="$(json_decision "$S1_GATE_EVAL/eval_result.json")"
echo "[Stage 1] pilot Gate-val: $GATE_DECISION"
if [[ "$GATE_DECISION" != "GO" && "$ALLOW_STAGE1_GATE_NO_GO" != 1 ]]; then
  die "Stage-1 Gate-val is $GATE_DECISION; rerun with --allow-stage1-gate-no-go for exploratory continuation"
fi

S1_CHECKPOINT="$S1_PILOT_TRAIN/$S1_CHECKPOINT_SELECTION.pt"
S1_CHECKPOINT_UPDATE="$("$PYTHON_BIN" - "$S1_CHECKPOINT" "$S1_CHECKPOINT_SELECTION" "$S1_PILOT_UPDATES" <<'PY'
import sys
import torch

path, selection, expected_updates = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
observed = int(checkpoint["optimizer_update"])
if selection == "last" and observed != int(expected_updates):
    raise SystemExit(
        f"Stage-1 last.pt update mismatch: {observed} != {expected_updates}"
    )
print(observed)
PY
)"
echo "[Stage 2] initialization: Stage-1 $S1_CHECKPOINT_SELECTION.pt at update $S1_CHECKPOINT_UPDATE"
S1_CHECKPOINT_SHA="$("$PYTHON_BIN" - "$S1_CHECKPOINT" <<'PY'
import hashlib
import sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"

# ----------------------------- Stage 2 ------------------------------------
# Base candidate may intentionally exit non-zero when it finds non-EOS rows.  It
# has still persisted the complete immutable candidate cache; the EOS-pool
# repair consumes that cache and deterministically extends it as necessary.
CANDIDATE_STATUS=0
if complete_files "$S2_CANDIDATE" data_manifest.json stage2_records.jsonl sharer_drafts.jsonl prepare_result.json; then
  echo "[Stage 2] reuse completed base candidate cache"
else
  set +e
  run_logged "$OUTPUT_ROOT/logs/stage2_candidate.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/prepare_draft_kv_openhermes_stage2.py" \
      --data "$OPENHERMES" \
      --data-split train \
      --receiver "$RECEIVER" \
      --sharer "$SHARER" \
      --exclude-stage1-manifest "$S1_DATA/data_manifest.json" \
      --output-dir "$S2_CANDIDATE" \
      --layer-mapping "$LAYER_MAPPING" \
      --train-examples 8192 \
      --gate-examples 512 \
      --reserve-examples 512 \
      --min-gold-tokens 16 \
      --max-gold-tokens 256 \
      --max-receiver-length 1024 \
      --max-receiver-text-length 3072 \
      --max-sharer-prompt-tokens 1536 \
      --max-sharer-length 2048 \
      --draft-max-new-tokens 512 \
      --draft-batch-size "$S2_DRAFT_BATCH" \
      --seed 91827 \
      --derangement-seed 71031 \
      --device "$DEVICE"
  CANDIDATE_STATUS=$?
  set -e
  if (( CANDIDATE_STATUS != 0 )); then
    candidate_cache_ready || die "Stage-2 candidate preparation failed before a complete cache was written; inspect stage2_candidate.log"
    echo "[Stage 2] candidate returned $CANDIDATE_STATUS after producing a complete cache; continuing with EOS-pool repair"
  fi
fi

if complete_files "$S2_EOS" data_manifest.json stage2_records.jsonl sharer_drafts.jsonl prepare_result.json; then
  echo "[Stage 2] reuse completed EOS-pool data"
else
  run_logged "$OUTPUT_ROOT/logs/stage2_eos_pool.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/prepare_draft_kv_openhermes_stage2_eos_pool.py" \
      --candidate-dir "$S2_CANDIDATE" \
      --output-dir "$S2_EOS" \
      --draft-batch-size "$S2_DRAFT_BATCH" \
      --draft-max-new-tokens 512 \
      --train-examples 8192 \
      --gate-examples 512 \
      --reserve-examples 512 \
      --derangement-seed 20260829 \
      --device "$DEVICE"
fi

if complete_files "$S2_TRAIN" last.pt train_history.json train_result.json; then
  if [[ -f "$S2_TRAIN/best.pt" ]]; then
    echo "[Stage 2] reuse completed 4,000-update OpenHermes answer training (best.pt available)"
  else
    echo "[Stage 2] reuse completed 4,000-update OpenHermes answer training (last.pt only; no eligible best.pt)"
  fi
else
  run_logged "$OUTPUT_ROOT/logs/stage2_train.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/train_draft_kv_openhermes_stage2.py" \
      --data-dir "$S2_EOS" \
      --stage1-data-dir "$S1_DATA" \
      --stage1-checkpoint "$S1_CHECKPOINT" \
      --expected-stage1-checkpoint-sha256 "$S1_CHECKPOINT_SHA" \
      --output-dir "$S2_TRAIN" \
      --max-updates 4000 \
      --microbatch "$S2_MICROBATCH" \
      --grad-accum "$S2_GRAD_ACCUM" \
      --eval-batch-size 1 \
      --reconstruction-replay-every 5 \
      --projector-lr 2e-4 \
      --gate-lr 1e-3 \
      --eval-every 250 \
      --log-every 10 \
      --max-reconstruction-nll-increase 0.10 \
      --min-reconstruction-gap-fraction 0.90 \
      --seed 91827 \
      --device "$DEVICE"
fi

# ----------------------------- Stage 3 ------------------------------------
if complete_files "$S3_DATA" data_manifest.json mc_records.jsonl sharer_drafts.jsonl prepare_result.json; then
  echo "[Stage 3] reuse completed ARC train/calibration data"
else
  run_logged "$OUTPUT_ROOT/logs/stage3_prepare.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/prepare_draft_kv_mc_train.py" \
      --data-root "$DATA_ROOT" \
      --receiver "$RECEIVER" \
      --sharer "$SHARER" \
      --layer-mapping "$LAYER_MAPPING" \
      --output-dir "$S3_DATA" \
      --draft-batch-size "$S3_DRAFT_BATCH" \
      --draft-max-new-tokens 512 \
      --max-receiver-length 1280 \
      --max-sharer-length 1792 \
      --device "$DEVICE"
fi

if complete_files "$S3_TRAIN" last.pt train_history.json train_result.json; then
  echo "[Stage 3] reuse completed 4,000-update ARC option training"
else
  run_logged "$OUTPUT_ROOT/logs/stage3_train.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/train_stage3.py" \
      --data-dir "$S3_DATA" \
      --reconstruction-data-dir "$S1_DATA" \
      --initial-stage2-checkpoint "$S2_TRAIN/last.pt" \
      --output-dir "$S3_TRAIN" \
      --max-mc-updates 4000 \
      --mc-updates-per-replay 4 \
      --microbatch "$S3_MICROBATCH" \
      --grad-accum "$S3_GRAD_ACCUM" \
      --eval-batch-size 1 \
      --projector-lr 1e-4 \
      --gate-lr 5e-4 \
      --protection-weight 0.10 \
      --protection-tolerance 0.10 \
      --eval-every-mc-updates 250 \
      --log-every-mc-updates 10 \
      --max-reconstruction-nll-increase 0.10 \
      --min-reconstruction-gap-fraction 0.90 \
      --seed 31847 \
      --device "$DEVICE"
fi

echo
echo "Three-stage training finished."
echo "  Stage-1 initialization: $S1_CHECKPOINT (update $S1_CHECKPOINT_UPDATE)"
echo "  Stage-2 last: $S2_TRAIN/last.pt"
echo "  Stage-3 last: $S3_TRAIN/last.pt"
echo "  Stage-3 best checkpoint: $S3_TRAIN/best.pt"
echo "  Logs and manifests: $OUTPUT_ROOT"
