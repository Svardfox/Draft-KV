"""Prepare OpenHermes Stage-2 splits and cache exact Sharer response tokens.

The JSONL cache is atomically rewritten after each batch and is resumable.
Split groups are generated independently so padding neighbours never cross a
registered split boundary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_openhermes_stage2_data import (  # noqa: E402
    PROTOCOL,
    TEXT_DRAFT_HEADER,
    build_stage2_record,
    encode_stage2_record,
    read_jsonl_by_id,
    receiver_target_encoding,
    sharer_prompt_token_ids,
    validate_draft_row,
    write_jsonl,
)
from draft_kv.train.draft_kv_reconstruction_data import (  # noqa: E402
    make_no_fixed_point_id_mapping,
)
from draft_kv.train.openhermes import load_openhermes_records  # noqa: E402
from script.draft_kv.draft_kv_common import (  # noqa: E402
    load_causal_lm,
    parse_layer_mapping,
    prepare_tokenizer,
    seed_all,
    sha256_file,
    write_json,
)


def _load_excluded_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_ids = manifest.get("split_ids")
    if not isinstance(split_ids, Mapping):
        raise RuntimeError("exclusion manifest has no split_ids mapping")
    return {
        str(example_id)
        for values in split_ids.values()
        for example_id in values
    }


def _trim_generated(
    values: Sequence[int], *, eos_ids: set[int], pad_token_id: int
) -> tuple[list[int], bool]:
    result: list[int] = []
    terminated = False
    for value in values:
        token = int(value)
        result.append(token)
        if token in eos_ids:
            terminated = True
            break
        if token == int(pad_token_id) and token not in eos_ids:
            result.pop()
            break
    return result, terminated


def _command_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "data": str(Path(args.data).resolve()) if Path(args.data).exists() else args.data,
        "data_split": str(args.data_split),
        "receiver": str(Path(args.receiver).resolve()),
        "sharer": str(Path(args.sharer).resolve()),
        "layer_mapping": {
            str(target): int(source)
            for target, source in parse_layer_mapping(args.layer_mapping).items()
        },
        "stage1_manifest": str(Path(args.exclude_stage1_manifest).resolve()),
        "train_examples": int(args.train_examples),
        "gate_examples": int(args.gate_examples),
        "reserve_examples": int(args.reserve_examples),
        "seed": int(args.seed),
        "derangement_seed": int(args.derangement_seed),
        "min_gold_tokens": int(args.min_gold_tokens),
        "max_gold_tokens": int(args.max_gold_tokens),
        "max_receiver_length": int(args.max_receiver_length),
        "max_receiver_text_length": int(args.max_receiver_text_length),
        "max_sharer_prompt_tokens": int(args.max_sharer_prompt_tokens),
        "max_sharer_length": int(args.max_sharer_length),
        "draft_batch_size": int(args.draft_batch_size),
        "draft_max_new_tokens": int(args.draft_max_new_tokens),
        "require_eos": True,
    }


def _semantic_command_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fields that determine records or generation semantics.

    Draft batch size is an execution-only choice. A resumed cache may use a
    different batch for its missing rows; the exact resulting cache remains
    authenticated by its content hash.
    """
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


def _prepare_records(
    args: argparse.Namespace,
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    source = load_openhermes_records(args.data, split=args.data_split)
    excluded = _load_excluded_ids(args.exclude_stage1_manifest)
    required = int(args.train_examples + args.gate_examples + args.reserve_examples)
    order = list(range(len(source)))
    random.Random(int(args.seed)).shuffle(order)
    selected: list[Dict[str, Any]] = []
    seen_records: set[str] = set()
    rejected: Counter[str] = Counter()
    scanned = 0
    for scanned, dataset_index in enumerate(order, start=1):
        example_id = f"openhermes:{dataset_index}"
        if example_id in excluded:
            rejected["excluded_stage1"] += 1
            continue
        try:
            row = build_stage2_record(source[dataset_index], dataset_index=dataset_index)
            if row["record_sha256"] in seen_records:
                raise ValueError("duplicate normalized context and gold")
            receiver_full, receiver_prompt_length, receiver_target = (
                receiver_target_encoding(receiver_tokenizer, row)
            )
            sharer_prompt = sharer_prompt_token_ids(sharer_tokenizer, row)
            if not int(args.min_gold_tokens) <= len(receiver_target) <= int(
                args.max_gold_tokens
            ):
                raise ValueError("gold token length")
            if len(receiver_full) > int(args.max_receiver_length):
                raise ValueError("Receiver sequence length")
            if len(sharer_prompt) > int(args.max_sharer_prompt_tokens):
                raise ValueError("Sharer prompt length")
            if len(sharer_prompt) + int(args.draft_max_new_tokens) > int(
                args.max_sharer_length
            ):
                raise ValueError("reserved Sharer sequence length")
        except (TypeError, ValueError) as error:
            rejected[str(error).split(":", 1)[0]] += 1
            continue
        seen_records.add(str(row["record_sha256"]))
        row.update(
            {
                "receiver_prompt_tokens": int(receiver_prompt_length),
                "receiver_gold_tokens": len(receiver_target),
                "receiver_sequence_tokens": len(receiver_full),
                "sharer_prompt_tokens": len(sharer_prompt),
            }
        )
        selected.append(row)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(
            f"only {len(selected)} qualifying Stage 2 records; required {required}; "
            f"rejections={dict(rejected)}"
        )
    counts = {
        "train": int(args.train_examples),
        "gate_val": int(args.gate_examples),
        "reserve_test": int(args.reserve_examples),
    }
    cursor = 0
    split_ids: Dict[str, list[str]] = {}
    for split, count in counts.items():
        rows = selected[cursor : cursor + count]
        for row in rows:
            row["split"] = split
        split_ids[split] = [str(row["example_id"]) for row in rows]
        cursor += count
    manifest_extra = {
        "counts": counts,
        "split_ids": split_ids,
        "gate_derangement": make_no_fixed_point_id_mapping(
            split_ids["gate_val"], seed=int(args.derangement_seed)
        ),
        "reserve_derangement": make_no_fixed_point_id_mapping(
            split_ids["reserve_test"], seed=int(args.derangement_seed) + 1
        ),
        "static_donor_id": split_ids["train"][0],
        "selection": {
            "scanned_records": scanned,
            "qualified_records": len(selected),
            "excluded_stage1_count": len(excluded),
            "rejections": dict(sorted(rejected.items())),
        },
    }
    return selected, manifest_extra


@torch.no_grad()
def generate_missing(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, Dict[str, Any]]:
    existing = read_jsonl_by_id(output_path)
    record_by_id = {str(row["example_id"]): row for row in records}
    if not set(existing).issubset(record_by_id):
        raise RuntimeError("resumed draft cache contains unregistered example IDs")
    for example_id, row in existing.items():
        validate_draft_row(row, record_by_id[example_id], tokenizer)
        if str(row.get("split")) != str(record_by_id[example_id]["split"]):
            raise RuntimeError("resumed draft cache contains a split mismatch")
    by_split: Dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        by_split.setdefault(str(row["split"]), []).append(row)
    eos_value = tokenizer.eos_token_id
    eos_ids = (
        {int(value) for value in eos_value}
        if isinstance(eos_value, (tuple, list, set))
        else ({int(eos_value)} if eos_value is not None else set())
    )
    total = len(records)
    completed = len(existing)
    for split in ("train", "gate_val", "reserve_test"):
        group = by_split.get(split, [])
        missing = [row for row in group if str(row["example_id"]) not in existing]
        for start in range(0, len(missing), int(batch_size)):
            chunk = missing[start : start + int(batch_size)]
            prompt_rows = [sharer_prompt_token_ids(tokenizer, row) for row in chunk]
            encoded = tokenizer.pad(
                {"input_ids": prompt_rows},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_rows: list[Dict[str, Any]] = []
            for position, record in enumerate(chunk):
                draft_ids, terminated = _trim_generated(
                    generated[position, input_width:].tolist(),
                    eos_ids=eos_ids,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                response = tokenizer.decode(draft_ids, skip_special_tokens=True).strip()
                if not draft_ids or not response:
                    raise RuntimeError(
                        f"Sharer generated an empty response for {record['example_id']}"
                    )
                row = {
                    "example_id": str(record["example_id"]),
                    "dataset_index": int(record["dataset_index"]),
                    "split": str(record["split"]),
                    "record_sha256": str(record["record_sha256"]),
                    "prompt_token_ids": prompt_rows[position],
                    "draft_token_ids": draft_ids,
                    "terminated_eos": bool(terminated),
                    "response": response,
                }
                new_rows.append(row)
                existing[str(row["example_id"])] = row
            ordered = [
                existing[str(row["example_id"])]
                for row in records
                if str(row["example_id"]) in existing
            ]
            write_jsonl(output_path, ordered)
            completed += len(new_rows)
            print(f"drafts {completed}/{total} split={split}", flush=True)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--sharer", required=True)
    parser.add_argument("--exclude-stage1-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-mapping", default="14:18,16:20,18:22,20:24")
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--gate-examples", type=int, default=512)
    parser.add_argument("--reserve-examples", type=int, default=512)
    parser.add_argument("--min-gold-tokens", type=int, default=16)
    parser.add_argument("--max-gold-tokens", type=int, default=256)
    parser.add_argument("--max-receiver-length", type=int, default=1024)
    parser.add_argument("--max-receiver-text-length", type=int, default=3072)
    parser.add_argument("--max-sharer-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-sharer-length", type=int, default=2048)
    parser.add_argument("--draft-max-new-tokens", type=int, default=512)
    parser.add_argument("--draft-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--derangement-seed", type=int, default=71031)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(
        args.train_examples,
        args.gate_examples,
        args.reserve_examples,
        args.min_gold_tokens,
        args.max_gold_tokens,
        args.max_receiver_length,
        args.max_receiver_text_length,
        args.max_sharer_prompt_tokens,
        args.max_sharer_length,
        args.draft_max_new_tokens,
        args.draft_batch_size,
    ) <= 0:
        raise ValueError("all Stage 2 sizes and lengths must be positive")
    seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "data_manifest.json"
    records_path = output / "stage2_records.jsonl"
    config = _command_config(args)

    receiver_tokenizer = prepare_tokenizer(args.receiver, padding_side="right")
    sharer_tokenizer = prepare_tokenizer(args.sharer, padding_side="left")
    if manifest_path.exists() or records_path.exists():
        if not (manifest_path.exists() and records_path.exists()):
            raise RuntimeError("partial Stage 2 preparation metadata; use a new directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_config = manifest.get("command_config", {})
        if manifest.get("protocol") != PROTOCOL or _semantic_command_config(
            existing_config
        ) != _semantic_command_config(config):
            raise RuntimeError("existing Stage 2 manifest differs from this command")
        if sha256_file(records_path) != manifest.get("records_sha256"):
            raise RuntimeError("Stage 2 records SHA differs from manifest")
        records = list(read_jsonl_by_id(records_path).values())
        if _record_draft_batch_size(manifest, args.draft_batch_size):
            write_json(manifest_path, manifest)
    else:
        records, extra = _prepare_records(args, receiver_tokenizer, sharer_tokenizer)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(records_path, records)
        manifest = {
            "protocol": PROTOCOL,
            "command_config": config,
            "draft_batch_sizes_used": [int(args.draft_batch_size)],
            "receiver": config["receiver"],
            "sharer": config["sharer"],
            "layer_mapping": config["layer_mapping"],
            "stage1_manifest": config["stage1_manifest"],
            "stage1_manifest_sha256": sha256_file(args.exclude_stage1_manifest),
            "source_data_sha256": (
                sha256_file(args.data) if Path(args.data).is_file() else None
            ),
            "records_file": records_path.name,
            "records_sha256": sha256_file(records_path),
            "draft_cache_file": "sharer_drafts.jsonl",
            "text_control_header": TEXT_DRAFT_HEADER,
            "lengths": {
                "max_receiver_length": int(args.max_receiver_length),
                "max_receiver_text_length": int(args.max_receiver_text_length),
                "max_sharer_length": int(args.max_sharer_length),
            },
            **extra,
        }
        write_json(manifest_path, manifest)

    sharer = load_causal_lm(args.sharer, torch.device(args.device))
    drafts_path = output / str(manifest["draft_cache_file"])
    drafts = generate_missing(
        model=sharer,
        tokenizer=sharer_tokenizer,
        records=records,
        output_path=drafts_path,
        batch_size=args.draft_batch_size,
        max_new_tokens=args.draft_max_new_tokens,
    )
    missing = [row["example_id"] for row in records if row["example_id"] not in drafts]
    truncated = [
        row["example_id"] for row in drafts.values() if not row["terminated_eos"]
    ]
    if missing:
        raise RuntimeError(f"Stage 2 draft cache is missing {len(missing)} rows")
    if truncated:
        raise RuntimeError(
            f"{len(truncated)} Sharer drafts did not reach EOS; protocol forbids training"
        )
    observed_max = {
        "receiver_context_sequence_tokens": 0,
        "receiver_text_sequence_tokens": 0,
        "sharer_sequence_tokens": 0,
    }
    for record in records:
        draft = drafts[str(record["example_id"])]
        context_encoded = encode_stage2_record(
            record,
            draft,
            receiver_tokenizer,
            sharer_tokenizer,
            receiver_mode="context",
            max_receiver_length=args.max_receiver_length,
            max_sharer_length=args.max_sharer_length,
        )
        text_encoded = encode_stage2_record(
            record,
            draft,
            receiver_tokenizer,
            sharer_tokenizer,
            receiver_mode="text",
            max_receiver_length=args.max_receiver_text_length,
            max_sharer_length=args.max_sharer_length,
        )
        observed_max["receiver_context_sequence_tokens"] = max(
            observed_max["receiver_context_sequence_tokens"],
            len(context_encoded["receiver_input_ids"]),
        )
        observed_max["receiver_text_sequence_tokens"] = max(
            observed_max["receiver_text_sequence_tokens"],
            len(text_encoded["receiver_input_ids"]),
        )
        observed_max["sharer_sequence_tokens"] = max(
            observed_max["sharer_sequence_tokens"],
            len(context_encoded["sharer_input_ids"]),
        )
    split_summary: Dict[str, Any] = {}
    for split in ("train", "gate_val", "reserve_test"):
        rows = [drafts[row["example_id"]] for row in records if row["split"] == split]
        split_summary[split] = {
            "count": len(rows),
            "mean_draft_tokens": sum(len(row["draft_token_ids"]) for row in rows) / len(rows),
            "terminated_eos": sum(bool(row["terminated_eos"]) for row in rows),
        }
    result = {
        "protocol": PROTOCOL,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "records": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "draft_cache": str(drafts_path.resolve()),
        "draft_cache_sha256": sha256_file(drafts_path),
        "splits": split_summary,
        "truncated_draft_count": len(truncated),
        "observed_max_lengths": observed_max,
    }
    write_json(output / "prepare_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
