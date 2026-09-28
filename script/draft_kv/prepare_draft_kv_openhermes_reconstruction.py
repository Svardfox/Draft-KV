"""Prepare deterministic OpenHermes splits for Draft-KV text reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_reconstruction_data import (  # noqa: E402
    PROTOCOL,
    RECONSTRUCTION_SYSTEM,
    RECONSTRUCTION_USER,
    build_reconstruction_record,
    encode_reconstruction_record,
    make_no_fixed_point_id_mapping,
    write_reconstruction_records,
)
from draft_kv.train.openhermes import load_openhermes_records  # noqa: E402
from script.draft_kv.draft_kv_common import (  # noqa: E402
    parse_layer_mapping,
    prepare_tokenizer,
    sha256_file,
    write_json,
)


def _rejection_key(error: ValueError) -> str:
    text = str(error)
    known = (
        "unknown roles",
        "not a message sequence",
        "must end with assistant",
        "no user message",
        "empty",
        "already contains",
        "special token",
        "shorter than",
        "longer than",
        "Receiver reconstruction sequence",
        "Sharer source sequence",
        "exact token suffix",
    )
    for value in known:
        if value in text:
            return value.replace(" ", "_")
    return "other"


def _load_excluded_example_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"excluded manifest does not exist: {source}")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("excluded manifest has an unexpected protocol")
    return {
        str(example_id)
        for values in manifest.get("split_ids", {}).values()
        for example_id in values
    }


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output_dir)
    manifest_path = output / "data_manifest.json"
    records_path = output / "reconstruction_records.jsonl"
    if any(
        path.exists()
        for path in (manifest_path, records_path, output / "prepare_result.json")
    ):
        raise RuntimeError(
            "reconstruction preparation outputs already exist; use a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)

    receiver_tokenizer = prepare_tokenizer(args.receiver, padding_side="right")
    sharer_tokenizer = prepare_tokenizer(args.sharer, padding_side="right")
    source_records = load_openhermes_records(args.data, split=args.data_split)
    excluded_ids = _load_excluded_example_ids(args.exclude_manifest)
    required = int(args.train_examples + args.gate_examples + args.reserve_examples)
    if required <= 0 or required > len(source_records):
        raise ValueError("requested split size is invalid for the source dataset")
    if not 0 < int(args.overfit_examples) <= int(args.train_examples):
        raise ValueError("overfit_examples must be in [1, train_examples]")

    source_order = list(range(len(source_records)))
    random.Random(int(args.seed)).shuffle(source_order)
    selected: list[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    rejected: Counter[str] = Counter()
    receiver_prompt_ids: list[int] | None = None
    for dataset_index in source_order:
        candidate_id = f"openhermes:{int(dataset_index)}"
        if candidate_id in excluded_ids:
            rejected["excluded_by_prior_manifest"] += 1
            continue
        try:
            row = build_reconstruction_record(
                source_records[dataset_index],
                dataset_index=dataset_index,
                seed=args.seed,
                include_private_key=not args.no_private_key,
            )
            canonical = str(row["canonical_message_sha256"])
            if canonical in seen_hashes:
                rejected["duplicate_canonical_message"] += 1
                continue
            encoded = encode_reconstruction_record(
                row,
                receiver_tokenizer,
                sharer_tokenizer,
                min_message_tokens=args.min_message_tokens,
                max_message_tokens=args.max_message_tokens,
                max_receiver_length=args.max_receiver_length,
                max_sharer_length=args.max_sharer_length,
            )
        except ValueError as error:
            rejected[_rejection_key(error)] += 1
            continue
        seen_hashes.add(canonical)
        current_prompt_ids = [
            int(value) for value in encoded["receiver_prompt_input_ids"]
        ]
        if receiver_prompt_ids is None:
            receiver_prompt_ids = current_prompt_ids
        elif current_prompt_ids != receiver_prompt_ids:
            raise RuntimeError(
                "Receiver reconstruction prompt is not sample-independent"
            )
        row["receiver_message_tokens"] = int(encoded["receiver_message_tokens"])
        row["receiver_natural_message_tokens"] = int(
            encoded["receiver_natural_message_tokens"]
        )
        row["sharer_message_tokens"] = int(encoded["sharer_message_tokens"])
        row["receiver_sequence_tokens"] = len(encoded["receiver_input_ids"])
        row["sharer_sequence_tokens"] = len(encoded["sharer_input_ids"])
        selected.append(row)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(
            f"only {len(selected)} qualifying records found; required {required}; "
            f"rejections={dict(rejected)}"
        )
    private_keys = [str(row["private_key"]) for row in selected if row["private_key"]]
    if private_keys and len(private_keys) != len(set(private_keys)):
        raise RuntimeError("private reconstruction keys are not unique")
    if receiver_prompt_ids is None:
        raise RuntimeError("no Receiver reconstruction prompt was encoded")

    split_counts = {
        "train": int(args.train_examples),
        "gate_val": int(args.gate_examples),
        "reserve_test": int(args.reserve_examples),
    }
    cursor = 0
    split_ids: Dict[str, list[str]] = {}
    for split, count in split_counts.items():
        rows = selected[cursor : cursor + count]
        for row in rows:
            row["split"] = split
        split_ids[split] = [str(row["example_id"]) for row in rows]
        cursor += count
    train_rows = selected[: split_counts["train"]]
    overfit_candidates = [
        row
        for row in train_rows
        if int(row["receiver_message_tokens"])
        <= int(args.overfit_max_message_tokens)
    ]
    if len(overfit_candidates) < int(args.overfit_examples):
        raise RuntimeError(
            "not enough short train messages for the requested overfit subset"
        )
    overfit_ids = [
        str(row["example_id"])
        for row in overfit_candidates[: int(args.overfit_examples)]
    ]
    gate_derangement = make_no_fixed_point_id_mapping(
        split_ids["gate_val"], seed=int(args.derangement_seed)
    )
    reserve_derangement = make_no_fixed_point_id_mapping(
        split_ids["reserve_test"], seed=int(args.derangement_seed) + 1
    )
    static_donor_id = next(
        (
            example_id
            for example_id in split_ids["train"]
            if example_id not in set(overfit_ids)
        ),
        split_ids["train"][0],
    )

    write_reconstruction_records(records_path, selected)
    data_path = Path(args.data)
    source_sha256 = None
    if data_path.exists() and not args.skip_source_sha256:
        source_sha256 = sha256_file(data_path)
    manifest = {
        "protocol": PROTOCOL,
        "reconstruction_mode": str(args.reconstruction_mode),
        "receiver": str(Path(args.receiver).resolve()),
        "sharer": str(Path(args.sharer).resolve()),
        "layer_mapping": parse_layer_mapping(args.layer_mapping),
        "source_data": (
            str(data_path.resolve()) if data_path.exists() else str(args.data)
        ),
        "source_data_split": str(args.data_split),
        "source_data_sha256": source_sha256,
        "source_data_size_bytes": (
            data_path.stat().st_size if data_path.exists() else None
        ),
        "excluded_manifest": (
            str(Path(args.exclude_manifest).resolve())
            if args.exclude_manifest
            else None
        ),
        "excluded_example_count": len(excluded_ids),
        "records_file": records_path.name,
        "records_sha256": sha256_file(records_path),
        "seed": int(args.seed),
        "derangement_seed": int(args.derangement_seed),
        "private_key_enabled": not args.no_private_key,
        "reconstruction_prompt": {
            "system": RECONSTRUCTION_SYSTEM,
            "user": RECONSTRUCTION_USER,
            "token_count": len(receiver_prompt_ids),
            "receiver_token_ids_sha256": hashlib.sha256(
                json.dumps(receiver_prompt_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "overfit_max_message_tokens": int(args.overfit_max_message_tokens),
        "lengths": {
            "min_message_tokens": int(args.min_message_tokens),
            "max_message_tokens": int(args.max_message_tokens),
            "max_receiver_length": int(args.max_receiver_length),
            "max_sharer_length": int(args.max_sharer_length),
        },
        "counts": split_counts | {"overfit": len(overfit_ids)},
        "split_ids": split_ids,
        "overfit_ids": overfit_ids,
        "gate_derangement": gate_derangement,
        "reserve_derangement": reserve_derangement,
        "static_donor_id": static_donor_id,
        "selection": {
            "scanned_records": source_order.index(selected[-1]["dataset_index"]) + 1,
            "qualified_records": len(selected),
            "rejections": dict(sorted(rejected.items())),
            "deduplicate_by": "NFKC-casefold-whitespace canonical assistant message",
        },
    }
    write_json(manifest_path, manifest)
    result = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "records": str(records_path.resolve()),
        "records_sha256": manifest["records_sha256"],
        "counts": manifest["counts"],
        "selection": manifest["selection"],
    }
    write_json(output / "prepare_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--sharer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-mapping", default="14:18,16:20,18:22,20:24")
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--gate-examples", type=int, default=512)
    parser.add_argument("--reserve-examples", type=int, default=512)
    parser.add_argument("--overfit-examples", type=int, default=32)
    parser.add_argument("--overfit-max-message-tokens", type=int, default=64)
    parser.add_argument("--min-message-tokens", type=int, default=16)
    parser.add_argument("--max-message-tokens", type=int, default=128)
    parser.add_argument("--max-receiver-length", type=int, default=256)
    parser.add_argument("--max-sharer-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--derangement-seed", type=int, default=71031)
    parser.add_argument("--no-private-key", action="store_true")
    parser.add_argument("--skip-source-sha256", action="store_true")
    parser.add_argument("--reconstruction-mode", default="base")
    parser.add_argument("--exclude-manifest")
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
