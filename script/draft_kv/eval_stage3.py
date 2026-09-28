#!/usr/bin/env python3
"""Evaluate Stage 3 checkpoints with a hard calibration-before-test gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_mc_training_data import (  # noqa: E402
    DraftKVMCOptionCollator,
    DraftKVMCOptionDataset,
    EVAL_PROTOCOL,
    PROTOCOL,
    load_mc_bundle,
    make_no_fixed_point_mapping,
    normalize_arc_file,
    option_token_ids,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    build_model,
    load_trainable_state,
    seed_all,
)
from script.draft_kv.prepare_draft_kv_mc_train import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    generate_missing,
)
from script.draft_kv.stage3_common import mc_validation
from script.draft_kv.train_stage3 import TRAIN_PROTOCOL

SUPPORTED_TRAIN_PROTOCOLS = frozenset({TRAIN_PROTOCOL})


DEFAULT_MC_DATA = (
    "/workspace/draft-kv/"
    "stage3_mc_training/data"
)
DEFAULT_CHECKPOINT = (
    "/workspace/draft-kv/"
    "stage3_mc_training/train/best.pt"
)
DEFAULT_OUTPUT_ROOT = (
    "/workspace/draft-kv/"
    "stage3_mc_training/eval"
)


def _checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    records_path: Path,
    drafts_path: Path,
) -> None:
    if not str(checkpoint.get("protocol", "")).startswith(TRAIN_PROTOCOL):
        raise RuntimeError("checkpoint is not a Stage 3 training checkpoint")
    if not str(checkpoint.get("data_protocol", "")).startswith(PROTOCOL):
        raise RuntimeError("checkpoint uses an unexpected MC data protocol")
    expected = {
        "mc_data_manifest_sha256": sha256_file(manifest_path),
        "mc_records_sha256": sha256_file(records_path),
        "mc_drafts_sha256": sha256_file(drafts_path),
    }
    for key, value in expected.items():
        if str(checkpoint.get(key)) != value:
            raise RuntimeError(f"checkpoint {key} differs from prepared MC data")
    for name in ("receiver", "sharer"):
        if Path(str(checkpoint.get(name, ""))).resolve() != Path(
            str(manifest[name])
        ).resolve():
            raise RuntimeError(f"checkpoint {name} differs from MC manifest")
    observed_mapping = {
        int(target): int(source)
        for target, source in checkpoint.get("layer_mapping", {}).items()
    }
    expected_mapping = {
        int(target): int(source)
        for target, source in manifest["layer_mapping"].items()
    }
    if observed_mapping != expected_mapping:
        raise RuntimeError("checkpoint layer mapping differs from MC manifest")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)


def _arc_test_registration(data_root: Path) -> list[Dict[str, Any]]:
    manifest_path = data_root / "ai2_arc" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = []
    for dataset, config in (("arc-e", "ARC-Easy"), ("arc-c", "ARC-Challenge")):
        path = data_root / "ai2_arc" / "raw" / config / "test.jsonl"
        relative = str(path.relative_to(data_root / "ai2_arc"))
        matches = [
            dict(row)
            for row in manifest["datasets"][dataset]["files"]
            if str(row.get("path")) == relative
        ]
        if len(matches) != 1:
            raise RuntimeError(f"ARC manifest does not register {relative}")
        entry = matches[0]
        if str(entry["sha256"]) != sha256_file(path):
            raise RuntimeError(f"ARC test SHA mismatch: {relative}")
        result.append(
            {
                "dataset": dataset,
                "split": "test",
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "count": int(entry["count"]),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    return result


def _test_records(data_root: Path) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    registration = _arc_test_registration(data_root)
    rows: list[Dict[str, Any]] = []
    for entry in registration:
        source = normalize_arc_file(
            entry["path"], dataset=entry["dataset"], source_split="test"
        )
        if len(source) != int(entry["count"]):
            raise RuntimeError("normalized ARC test count differs from manifest")
        for row in source:
            row["task_split"] = "test"
        rows.extend(source)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["example_id"])))
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("ARC test IDs are not globally unique")
    return rows, registration


def _selection_gate(
    path: Path,
    *,
    checkpoint_sha256: str,
    mc_manifest_sha256: str,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "test evaluation requires a completed calibration selection result"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        not str(result.get("protocol", "")).startswith(EVAL_PROTOCOL)
        or result.get("split") != "calibration"
        or result.get("decision") != "SELECTED"
    ):
        raise RuntimeError("calibration prerequisite did not select a checkpoint")
    if str(result.get("checkpoint_sha256")) != str(checkpoint_sha256):
        raise RuntimeError("test checkpoint differs from selected checkpoint")
    if str(result.get("mc_data_manifest_sha256")) != str(mc_manifest_sha256):
        raise RuntimeError("selection used a different MC data manifest")
    eval_manifest_path = path.parent / "eval_manifest.json"
    if not eval_manifest_path.is_file():
        raise FileNotFoundError("selection has no eval_manifest.json")
    if str(result.get("eval_manifest_sha256")) != sha256_file(eval_manifest_path):
        raise RuntimeError("selection result does not authenticate its eval manifest")
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if (
        not str(eval_manifest.get("protocol", "")).startswith(EVAL_PROTOCOL)
        or eval_manifest.get("split") != "calibration"
        or str(eval_manifest.get("checkpoint_sha256")) != str(checkpoint_sha256)
        or str(eval_manifest.get("mc_data_manifest_sha256"))
        != str(mc_manifest_sha256)
    ):
        raise RuntimeError("selection eval manifest has inconsistent lineage")
    return {
        "selection_result": str(path.resolve()),
        "selection_result_sha256": sha256_file(path),
        "selection_eval_manifest": str(eval_manifest_path.resolve()),
        "selection_eval_manifest_sha256": sha256_file(eval_manifest_path),
    }


def _ensure_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != dict(value):
            raise RuntimeError(f"existing evaluation manifest differs: {path}")
        return
    write_json_atomic(path, value)


def _summary_without_rows(validation: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in validation.items() if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--data-dir", default=DEFAULT_MC_DATA)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-result")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--draft-batch-size", type=int, default=16)
    parser.add_argument("--draft-max-new-tokens", type=int, default=512)
    parser.add_argument("--derangement-seed", type=int, default=99173)
    parser.add_argument("--seed", type=int, default=31847)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(
        args.eval_batch_size,
        args.draft_batch_size,
        args.draft_max_new_tokens,
    ) <= 0:
        raise ValueError("batch sizes and generation length must be positive")
    seed_all(args.seed)

    (
        manifest,
        records,
        drafts,
        manifest_path,
        records_path,
        drafts_path,
    ) = load_mc_bundle(args.data_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _checkpoint_contract(
        checkpoint,
        checkpoint_path=checkpoint_path,
        manifest=manifest,
        manifest_path=manifest_path,
        records_path=records_path,
        drafts_path=drafts_path,
    )
    if args.split == "calibration":
        if not bool(
            checkpoint.get("reconstruction_preservation", {}).get("eligible")
        ):
            raise RuntimeError(
                "checkpoint is not eligible for selection because OpenHermes "
                "reconstruction preservation failed"
            )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(args.output_root).resolve() / args.split / checkpoint_sha[:12]
    )
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError("evaluation result exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)

    selection = None
    source_registration: list[Dict[str, Any]]
    if args.split == "calibration":
        evaluation_records = records
        evaluation_drafts = drafts
        derangement = manifest["calibration_derangement"]
        source_registration = list(manifest["source_registration"])
        draft_cache_path = drafts_path
    else:
        if not args.selection_result:
            raise ValueError("--selection-result is mandatory for test evaluation")
        selection = _selection_gate(
            Path(args.selection_result).resolve(),
            checkpoint_sha256=checkpoint_sha,
            mc_manifest_sha256=sha256_file(manifest_path),
        )
        evaluation_records, source_registration = _test_records(
            Path(args.data_root).resolve()
        )
        test_ids = [str(row["example_id"]) for row in evaluation_records]
        derangement = make_no_fixed_point_mapping(
            test_ids, seed=args.derangement_seed
        )
        draft_cache_path = output / "sharer_drafts.jsonl"
        evaluation_drafts = {}

    eval_manifest = {
        "protocol": EVAL_PROTOCOL,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "mc_data_manifest": str(manifest_path.resolve()),
        "mc_data_manifest_sha256": sha256_file(manifest_path),
        "receiver": str(manifest["receiver"]),
        "sharer": str(manifest["sharer"]),
        "layer_mapping": dict(manifest["layer_mapping"]),
        "conditions": ["Zero", "Matched", "Deranged"],
        "option_scoring": "gold next-token CE over valid space-prefixed A-J tokens",
        "source_registration": source_registration,
        "derangement": derangement,
        "selection_prerequisite": selection,
        "configuration": {
            "eval_batch_size": int(args.eval_batch_size),
            "draft_batch_size": int(args.draft_batch_size),
            "draft_max_new_tokens": int(args.draft_max_new_tokens),
            "derangement_seed": int(args.derangement_seed),
            "seed": int(args.seed),
        },
    }
    _ensure_json(output / "eval_manifest.json", eval_manifest)

    model, receiver_tokenizer, sharer_tokenizer = build_model(
        receiver_path=str(manifest["receiver"]),
        sharer_path=str(manifest["sharer"]),
        layer_mapping=manifest["layer_mapping"],
        device_name=args.device,
    )
    load_trainable_state(model, checkpoint)
    model.set_stage("eval")
    if model.trainable_parameter_names():
        raise RuntimeError("evaluation model still has trainable parameters")

    if args.split == "test":
        evaluation_drafts = generate_missing(
            model=model.sharer,
            tokenizer=sharer_tokenizer,
            records=evaluation_records,
            output_path=draft_cache_path,
            batch_size=args.draft_batch_size,
            max_new_tokens=args.draft_max_new_tokens,
            max_sharer_length=int(manifest["lengths"]["max_sharer_length"]),
        )
        if set(evaluation_drafts) != {
            str(row["example_id"]) for row in evaluation_records
        }:
            raise RuntimeError("ARC test Sharer cache is incomplete")

    dataset = DraftKVMCOptionDataset(
        evaluation_records,
        evaluation_drafts,
        receiver_tokenizer,
        sharer_tokenizer,
        split=args.split,
        max_receiver_length=int(manifest["lengths"]["max_receiver_length"]),
        max_sharer_length=int(manifest["lengths"]["max_sharer_length"]),
    )
    collator = DraftKVMCOptionCollator(
        receiver_tokenizer, sharer_tokenizer
    )
    validation = mc_validation(
        model,
        dataset,
        collator,
        derangement=derangement,
        option_ids=option_token_ids(receiver_tokenizer),
        batch_size=args.eval_batch_size,
    )
    summary = _summary_without_rows(validation)
    if any(
        not math.isfinite(float(values[key]))
        for values in summary["conditions"].values()
        for key in ("accuracy", "mean_nll", "mean_gold_log_probability")
    ):
        raise RuntimeError("evaluation produced a non-finite metric")
    write_jsonl_atomic(output / "per_example.jsonl", validation["rows"])
    result = {
        "protocol": EVAL_PROTOCOL,
        "split": args.split,
        "decision": "SELECTED" if args.split == "calibration" else "FINAL_REPORT",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "mc_data_manifest_sha256": sha256_file(manifest_path),
        "eval_manifest_sha256": sha256_file(output / "eval_manifest.json"),
        "per_example_sha256": sha256_file(output / "per_example.jsonl"),
        "draft_cache": str(draft_cache_path.resolve()),
        "draft_cache_sha256": sha256_file(draft_cache_path),
        "metrics": summary,
        "selection_prerequisite": selection,
        "test_policy": (
            "No ARC test rows were used."
            if args.split == "calibration"
            else "ARC test was opened only after authenticating SELECTED calibration."
        ),
    }
    write_json_atomic(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
