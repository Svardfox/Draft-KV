#!/usr/bin/env bash
# Run the complete three-stage Draft-KV training pipeline.
#
# Stage 1: OpenHermes textual reconstruction (overfit gate + long pilot)
# Stage 2: OpenHermes answer training with reconstruction replay
# Stage 3: ARC-E/ARC-C multiple-choice option training with replay
#
# The wrapper deliberately keeps every stage in a separate directory.  A
# completed stage is reused on rerun; all underlying programs still authenticate
# manifests, model paths, layer mappings and checkpoint lineage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${DRAFT_KV_PYTHON:-python3}"

# Reverse direction requested for this run: Qwen2.5 is the Sharer and Qwen3 is
# the Receiver.  Mapping syntax is Receiver-layer:Sharer-layer.
RECEIVER="/workspace/models/Qwen3-0.6B"
SHARER="/workspace/models/Qwen2.5-0.5B-Instruct"
LAYER_MAPPING="18:14,20:16,22:18,24:20"
OPENHERMES="/workspace/datasets/OpenHermes-2.5-500k/openhermes2_5_500k.json"
DATA_ROOT="/workspace/datasets"
OUTPUT_ROOT="/workspace/draft-kv/three_stage_qwen25_sharer_to_qwen3_receiver"
GPU_ID="0"
ALLOW_STAGE1_GATE_NO_GO=0

usage() {
  cat <<'EOF'
Usage:
  bash script/draft_kv/run_three_stage_training.sh [options]

Runs Stage 1 reconstruction, Stage 2 OpenHermes answer training, and Stage 3
ARC option training in one pipeline.  The default pair is:
  Sharer   Qwen2.5-0.5B-Instruct (24 layers)
  Receiver Qwen3-0.6B          (28 layers)
  Mapping  18:14,20:16,22:18,24:20 (Receiver:Sharer)

Options:
  --receiver PATH              Receiver model directory
  --sharer PATH                Sharer model directory
  --layer-mapping MAP          Receiver:Sharer pairs, e.g. 18:14,20:16
  --openhermes PATH            OpenHermes JSON/JSONL source
  --data-root PATH             Root containing ai2_arc/ for Stage 3
  --output-root PATH           Root for all data, checkpoints and logs
  --gpu-id ID                  Physical GPU exposed as cuda:0 (default: 0)
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
    --receiver)
      need_value "$@"
      RECEIVER="$2"
      shift 2
      ;;
    --sharer)
      need_value "$@"
      SHARER="$2"
      shift 2
      ;;
    --layer-mapping)
      need_value "$@"
      LAYER_MAPPING="$2"
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

[[ -x "$PYTHON_BIN" ]] || die "Python executable is not executable: $PYTHON_BIN"
[[ -f "$OPENHERMES" ]] || die "OpenHermes source does not exist: $OPENHERMES"
[[ -f "$RECEIVER/config.json" ]] || die "Receiver config is missing: $RECEIVER/config.json"
[[ -f "$SHARER/config.json" ]] || die "Sharer config is missing: $SHARER/config.json"
[[ -f "$DATA_ROOT/ai2_arc/manifest.json" ]] || die "ARC manifest is missing: $DATA_ROOT/ai2_arc/manifest.json"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
mkdir -p "$OUTPUT_ROOT/logs"

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
print(json.load(open(sys.argv[1], encoding="utf-8")).get("decision", ""))
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
echo "  Sharer:   $SHARER"
echo "  Receiver: $RECEIVER"
echo "  Mapping:  $LAYER_MAPPING"
echo "  Output:   $OUTPUT_ROOT"
echo "  GPU:      physical $GPU_ID -> $DEVICE"

# Persist an immutable orchestration record.  A rerun with a different pair or
# mapping on the same output root is rejected instead of silently mixing data.
"$PYTHON_BIN" - "$OUTPUT_ROOT/run_manifest.json" "$RECEIVER" "$SHARER" "$LAYER_MAPPING" "$OPENHERMES" "$DATA_ROOT" "$GPU_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "protocol": "draft_kv_three_stage_training",
    "receiver": str(Path(sys.argv[2]).resolve()),
    "sharer": str(Path(sys.argv[3]).resolve()),
    "layer_mapping": {
        str(int(item.split(":", 1)[0])): int(item.split(":", 1)[1])
        for item in sys.argv[4].split(",")
    },
    "openhermes": str(Path(sys.argv[5]).resolve()),
    "data_root": str(Path(sys.argv[6]).resolve()),
    "gpu_id": str(sys.argv[7]),
    "stage1_mode": "longrun",
    "stage1_pilot_updates": 16000,
    "stage2_updates": 4000,
    "stage3_mc_updates": 4000,
}
if path.exists():
    old = json.loads(path.read_text(encoding="utf-8"))
    if old != payload:
        raise SystemExit(
            f"existing run_manifest.json differs from this command: {path}"
        )
else:
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
  echo "[Stage 1] reuse completed Overfit-32 training"
else
  run_logged "$OUTPUT_ROOT/logs/stage1_overfit_train.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/train_draft_kv_openhermes_reconstruction.py" \
      --data-dir "$S1_DATA" \
      --output-dir "$S1_OVERFIT_TRAIN" \
      --mode overfit \
      --max-updates 300 \
      --projector-lr 1e-3 \
      --gate-lr 1e-2 \
      --microbatch 8 \
      --grad-accum 4 \
      --eval-batch-size 16 \
      --eval-every 50 \
      --log-every 10 \
      --seed 91827 \
      --device "$DEVICE"
fi

if complete_files "$S1_OVERFIT_EVAL" eval_result.json eval_manifest.json per_example_nll.jsonl per_example_generation.jsonl; then
  echo "[Stage 1] reuse completed Overfit-32 evaluation"
else
  run_logged "$OUTPUT_ROOT/logs/stage1_overfit_eval.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/eval_draft_kv_openhermes_reconstruction.py" \
      --data-dir "$S1_DATA" \
      --checkpoint "$S1_OVERFIT_TRAIN/last.pt" \
      --output-dir "$S1_OVERFIT_EVAL" \
      --split overfit \
      --batch-size 16 \
      --generation-examples 32 \
      --seed 91827 \
      --device "$DEVICE"
fi

OVERFIT_DECISION="$(json_decision "$S1_OVERFIT_EVAL/eval_result.json")"
[[ "$OVERFIT_DECISION" == "GO" ]] || die "Stage-1 Overfit gate is $OVERFIT_DECISION; pilot cannot start"
echo "[Stage 1] Overfit gate: GO"

if complete_files "$S1_PILOT_TRAIN" last.pt best.pt train_history.json train_result.json; then
  echo "[Stage 1] reuse completed 16,000-update pilot"
else
  run_logged "$OUTPUT_ROOT/logs/stage1_pilot_train.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/train_draft_kv_openhermes_reconstruction.py" \
      --data-dir "$S1_DATA" \
      --output-dir "$S1_PILOT_TRAIN" \
      --mode pilot \
      --max-updates 16000 \
      --projector-lr 1e-3 \
      --gate-lr 1e-2 \
      --microbatch 16 \
      --grad-accum 4 \
      --eval-batch-size 16 \
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
      --batch-size 16 \
      --generation-examples 64 \
      --seed 91827 \
      --device "$DEVICE"
fi

GATE_DECISION="$(json_decision "$S1_GATE_EVAL/eval_result.json")"
echo "[Stage 1] pilot Gate-val: $GATE_DECISION"
if [[ "$GATE_DECISION" != "GO" && "$ALLOW_STAGE1_GATE_NO_GO" != 1 ]]; then
  die "Stage-1 Gate-val is $GATE_DECISION; rerun with --allow-stage1-gate-no-go for exploratory continuation"
fi

S1_CHECKPOINT="$S1_PILOT_TRAIN/best.pt"
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
      --draft-batch-size 16 \
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
      --draft-batch-size 16 \
      --draft-max-new-tokens 512 \
      --train-examples 8192 \
      --gate-examples 512 \
      --reserve-examples 512 \
      --derangement-seed 20260829 \
      --device "$DEVICE"
fi

if complete_files "$S2_TRAIN" last.pt best.pt train_history.json train_result.json; then
  echo "[Stage 2] reuse completed 4,000-update OpenHermes answer training"
else
  run_logged "$OUTPUT_ROOT/logs/stage2_train.log" \
    "$PYTHON_BIN" "$REPO_ROOT/script/draft_kv/train_draft_kv_openhermes_stage2.py" \
      --data-dir "$S2_EOS" \
      --stage1-data-dir "$S1_DATA" \
      --stage1-checkpoint "$S1_CHECKPOINT" \
      --expected-stage1-checkpoint-sha256 "$S1_CHECKPOINT_SHA" \
      --output-dir "$S2_TRAIN" \
      --max-updates 4000 \
      --microbatch 2 \
      --grad-accum 16 \
      --eval-batch-size 8 \
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
      --draft-batch-size 16 \
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
      --microbatch 2 \
      --grad-accum 8 \
      --eval-batch-size 8 \
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
echo "  Stage-1 best: $S1_CHECKPOINT"
echo "  Stage-2 last: $S2_TRAIN/last.pt"
echo "  Stage-3 last: $S3_TRAIN/last.pt"
echo "  Stage-3 best (when preservation-eligible): $S3_TRAIN/best.pt"
echo "  Logs and manifests: $OUTPUT_ROOT"
