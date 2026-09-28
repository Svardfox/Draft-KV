#!/usr/bin/env python3
"""Prepare ARC train/calibration rows and a resumable Sharer draft cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_mc_training_data import (  # noqa: E402
    ARC_CONFIGS,
    PROTOCOL,
    TASK_SPLITS,
    deterministic_train_calibration_split,
    make_no_fixed_point_mapping,
    normalize_arc_file,
    read_jsonl,
    receiver_prompt_token_ids,
    sha256_file,
    sharer_prompt_token_ids,
    validate_draft_row,
    write_json_atomic,
    write_jsonl_atomic,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    load_causal_lm,
    parse_layer_mapping,
    prepare_tokenizer,
    seed_all,
)


DEFAULT_DATA_ROOT = "/workspace/datasets"
DEFAULT_RECEIVER = (
    "/workspace/models/Qwen2.5-0.5B-Instruct"
)
DEFAULT_SHARER = "/workspace/models/Qwen3-0.6B"
DEFAULT_MAPPING = "14:18,16:20,18:22,20:24"
DEFAULT_OUTPUT = (
    "/workspace/draft-kv/"
    "stage3_mc_training/data"
)


def _raw_path(root: Path, dataset: str, split: str) -> Path:
    return root / "ai2_arc" / "raw" / ARC_CONFIGS[dataset] / f"{split}.jsonl"


def _source_registration(
    data_root: Path, dataset: str, split: str, path: Path
) -> Dict[str, Any]:
    manifest_path = data_root / "ai2_arc" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(path.relative_to(data_root / "ai2_arc"))
    matches = [
        dict(row)
        for row in manifest["datasets"][dataset]["files"]
        if str(row.get("path")) == relative
    ]
    if len(matches) != 1:
        raise RuntimeError(f"ARC manifest does not uniquely register {relative}")
    entry = matches[0]
    observed = sha256_file(path)
    if str(entry.get("sha256")) != observed:
        raise RuntimeError(f"raw ARC source SHA mismatch: {relative}")
    return {
        "dataset": dataset,
        "split": split,
        "path": str(path.resolve()),
        "sha256": observed,
        "count": int(entry["count"]),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _trim_generated(
    values: Sequence[int], *, eos_ids: set[int], pad_token_id: int
) -> tuple[list[int], bool]:
    result: list[int] = []
    terminated = False
    for raw in values:
        token = int(raw)
        if token == int(pad_token_id) and token not in eos_ids:
            break
        result.append(token)
        if token in eos_ids:
            terminated = True
            break
    return result, terminated


def _ordered_cache(
    cache: Mapping[str, Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    return [
        cache[str(row["example_id"])]
        for row in records
        if str(row["example_id"]) in cache
    ]


@torch.no_grad()
def generate_missing(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    max_sharer_length: int,
) -> Dict[str, Dict[str, Any]]:
    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    cache = {str(row["example_id"]): row for row in existing_rows}
    if len(cache) != len(existing_rows):
        raise RuntimeError("resumed Sharer cache contains duplicate IDs")
    ids = [str(row["example_id"]) for row in records]
    cached_ids = [str(row["example_id"]) for row in existing_rows]
    if cached_ids != ids[: len(cached_ids)]:
        raise RuntimeError("resumed Sharer cache is not an ordered record prefix")
    if len(cached_ids) != len(ids) and len(cached_ids) % int(batch_size):
        raise RuntimeError("resumed Sharer cache ends inside a generation batch")
    by_id = {str(row["example_id"]): row for row in records}
    for example_id, row in cache.items():
        validate_draft_row(row, by_id[example_id], tokenizer)

    eos_value = tokenizer.eos_token_id
    eos_ids = (
        {int(value) for value in eos_value}
        if isinstance(eos_value, (tuple, list, set))
        else ({int(eos_value)} if eos_value is not None else set())
    )
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        missing = records[len(cache) :]
        for start in range(0, len(missing), int(batch_size)):
            chunk = missing[start : start + int(batch_size)]
            prompt_rows = [sharer_prompt_token_ids(tokenizer, row) for row in chunk]
            if max(len(row) + int(max_new_tokens) for row in prompt_rows) > int(
                max_sharer_length
            ):
                raise RuntimeError("reserved Sharer generation exceeds max length")
            encoded = tokenizer.pad(
                {"input_ids": prompt_rows},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            width = int(encoded["input_ids"].shape[1])
            generated = model.generate(
                **encoded,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=tokenizer.eos_token_id,
            )
            for position, record in enumerate(chunk):
                draft_ids, terminated = _trim_generated(
                    generated[position, width:].tolist(),
                    eos_ids=eos_ids,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                response = tokenizer.decode(
                    draft_ids, skip_special_tokens=True
                ).strip()
                if not draft_ids or not response:
                    raise RuntimeError(
                        f"empty Sharer draft for {record['example_id']}"
                    )
                cache[str(record["example_id"])] = {
                    "example_id": str(record["example_id"]),
                    "dataset": str(record["dataset"]),
                    "task_split": str(record["task_split"]),
                    "record_sha256": str(record["record_sha256"]),
                    "prompt_token_ids": prompt_rows[position],
                    "draft_token_ids": draft_ids,
                    "terminated_eos": bool(terminated),
                    "response": response,
                }
            write_jsonl_atomic(output_path, _ordered_cache(cache, records))
            print(f"Sharer cache {len(cache)}/{len(records)}", flush=True)
    finally:
        tokenizer.padding_side = previous_padding
    return cache


def _command_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "data_root": str(Path(args.data_root).resolve()),
        "receiver": str(Path(args.receiver).resolve()),
        "sharer": str(Path(args.sharer).resolve()),
        "layer_mapping": {
            str(target): int(source)
            for target, source in parse_layer_mapping(args.layer_mapping).items()
        },
        "calibration_fraction": float(args.calibration_fraction),
        "split_seed": int(args.split_seed),
        "derangement_seed": int(args.derangement_seed),
        "draft_batch_size": int(args.draft_batch_size),
        "draft_max_new_tokens": int(args.draft_max_new_tokens),
        "max_receiver_length": int(args.max_receiver_length),
        "max_sharer_length": int(args.max_sharer_length),
        "source_datasets": ["arc-e", "arc-c"],
        "source_splits": ["train", "validation"],
        "forbidden_training_splits": ["test"],
        "mmlu_redux_training": False,
    }


def _semantic_command_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Exclude execution-only batching from cache identity checks."""
    return {
        str(key): value
        for key, value in config.items()
        if str(key) != "draft_batch_size"
    }


def _record_draft_batch_size(
    manifest: Dict[str, Any], batch_size: int
) -> bool:
    had_history = "draft_batch_sizes_used" in manifest
    observed = {
        int(value) for value in manifest.get("draft_batch_sizes_used", [])
    }
    initial = manifest.get("command_config", {}).get("draft_batch_size")
    if initial is not None:
        observed.add(int(initial))
    before = sorted(observed)
    observed.add(int(batch_size))
    after = sorted(observed)
    manifest["draft_batch_sizes_used"] = after
    return after != before or not had_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--receiver", default=DEFAULT_RECEIVER)
    parser.add_argument("--sharer", default=DEFAULT_SHARER)
    parser.add_argument("--layer-mapping", default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=31847)
    parser.add_argument("--derangement-seed", type=int, default=71031)
    parser.add_argument("--draft-batch-size", type=int, default=16)
    parser.add_argument("--draft-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-receiver-length", type=int, default=1280)
    parser.add_argument("--max-sharer-length", type=int, default=1792)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(
        args.draft_batch_size,
        args.draft_max_new_tokens,
        args.max_receiver_length,
        args.max_sharer_length,
    ) <= 0:
        raise ValueError("batch sizes and sequence lengths must be positive")
    seed_all(args.split_seed)
    root = Path(args.data_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "data_manifest.json"
    records_path = output / "mc_records.jsonl"
    drafts_path = output / "sharer_drafts.jsonl"
    config = _command_config(args)

    receiver_tokenizer = prepare_tokenizer(args.receiver, padding_side="left")
    sharer_tokenizer = prepare_tokenizer(args.sharer, padding_side="left")
    if manifest_path.exists() or records_path.exists():
        if not (manifest_path.exists() and records_path.exists()):
            raise RuntimeError("partial MC preparation metadata; use a new directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != PROTOCOL or _semantic_command_config(
            manifest.get("command_config", {})
        ) != _semantic_command_config(config):
            raise RuntimeError("existing MC manifest differs from this command")
        if sha256_file(records_path) != str(manifest.get("records_sha256")):
            raise RuntimeError("existing MC records differ from manifest")
        records = read_jsonl(records_path)
        if _record_draft_batch_size(manifest, args.draft_batch_size):
            write_json_atomic(manifest_path, manifest)
    else:
        source_registration: list[Dict[str, Any]] = []
        source_rows: list[Dict[str, Any]] = []
        for dataset in ("arc-e", "arc-c"):
            for source_split in ("train", "validation"):
                path = _raw_path(root, dataset, source_split)
                if not path.is_file():
                    raise FileNotFoundError(f"missing raw ARC source: {path}")
                registration = _source_registration(
                    root, dataset, source_split, path
                )
                rows = normalize_arc_file(
                    path, dataset=dataset, source_split=source_split
                )
                if len(rows) != int(registration["count"]):
                    raise RuntimeError("raw ARC count differs from source manifest")
                source_registration.append(registration)
                source_rows.extend(rows)
        records = deterministic_train_calibration_split(
            source_rows,
            calibration_fraction=args.calibration_fraction,
            seed=args.split_seed,
        )

        rejected: Counter[str] = Counter()
        retained: list[Dict[str, Any]] = []
        for row in records:
            try:
                receiver_ids = receiver_prompt_token_ids(receiver_tokenizer, row)
                sharer_ids = sharer_prompt_token_ids(sharer_tokenizer, row)
                if len(receiver_ids) > int(args.max_receiver_length):
                    raise ValueError("Receiver prompt length")
                if len(sharer_ids) + int(args.draft_max_new_tokens) > int(
                    args.max_sharer_length
                ):
                    raise ValueError("reserved Sharer sequence length")
            except (TypeError, ValueError) as error:
                rejected[str(error).split(":", 1)[0]] += 1
                continue
            row["receiver_prompt_tokens"] = len(receiver_ids)
            row["sharer_prompt_tokens"] = len(sharer_ids)
            retained.append(row)
        records = retained
        split_ids = {
            split: [
                str(row["example_id"])
                for row in records
                if str(row["task_split"]) == split
            ]
            for split in TASK_SPLITS
        }
        if min(len(values) for values in split_ids.values()) < 2:
            raise RuntimeError("length filtering left an unusable MC split")
        write_jsonl_atomic(records_path, records)
        manifest = {
            "protocol": PROTOCOL,
            "command_config": config,
            "draft_batch_sizes_used": [int(args.draft_batch_size)],
            "receiver": config["receiver"],
            "sharer": config["sharer"],
            "layer_mapping": config["layer_mapping"],
            "records_file": records_path.name,
            "records_sha256": sha256_file(records_path),
            "draft_cache_file": drafts_path.name,
            "counts": {key: len(value) for key, value in split_ids.items()},
            "split_ids": split_ids,
            "train_derangement": make_no_fixed_point_mapping(
                split_ids["train"], seed=args.derangement_seed
            ),
            "calibration_derangement": make_no_fixed_point_mapping(
                split_ids["calibration"], seed=args.derangement_seed + 1
            ),
            "lengths": {
                "max_receiver_length": int(args.max_receiver_length),
                "max_sharer_length": int(args.max_sharer_length),
            },
            "source_registration": source_registration,
            "selection": {
                "source_count": len(source_rows),
                "retained_count": len(records),
                "rejections": dict(sorted(rejected.items())),
            },
            "test_policy": (
                "ARC-Easy test and ARC-Challenge test are absent from this "
                "bundle and reserved for post-selection final reporting."
            ),
            "mmlu_redux_policy": "MMLU-Redux is never loaded by this preparation.",
        }
        write_json_atomic(manifest_path, manifest)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    sharer = load_causal_lm(args.sharer, device)
    drafts = generate_missing(
        model=sharer,
        tokenizer=sharer_tokenizer,
        records=records,
        output_path=drafts_path,
        batch_size=args.draft_batch_size,
        max_new_tokens=args.draft_max_new_tokens,
        max_sharer_length=args.max_sharer_length,
    )
    if set(drafts) != {str(row["example_id"]) for row in records}:
        raise RuntimeError("Sharer cache is incomplete")
    truncated = [
        example_id
        for example_id, row in drafts.items()
        if not bool(row.get("terminated_eos"))
    ]
    result = {
        "protocol": PROTOCOL,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": str(records_path),
        "records_sha256": sha256_file(records_path),
        "draft_cache": str(drafts_path),
        "draft_cache_sha256": sha256_file(drafts_path),
        "counts": dict(manifest["counts"]),
        "truncated_draft_count": len(truncated),
        "truncated_example_ids": truncated,
    }
    write_json_atomic(output / "prepare_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
