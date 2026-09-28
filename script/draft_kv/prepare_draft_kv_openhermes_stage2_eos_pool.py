"""Build the Stage-2 EOS-pool mode from an immutable base candidate pool.

The original Stage-2 preparation registered 9,216 candidates before the
Sharer was run.  This protocol-level repair treats those rows as a candidate
pool, keeps only rows whose frozen Sharer response reached EOS, and continues
the exact deterministic candidate traversal only when more rows are needed.
Existing response token IDs are copied byte-for-byte; they are never
regenerated or filtered using gold answers, losses, or response quality.
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
    EOS_POOL_PROTOCOL,
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


def _load_excluded_ids(path: str | Path) -> set[str]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    values = manifest.get("split_ids")
    if not isinstance(values, Mapping):
        raise RuntimeError("Stage-1 exclusion manifest has no split_ids")
    return {str(value) for split in values.values() for value in split}


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


def _load_candidate_bundle(
    candidate_dir: str | Path,
) -> tuple[
    Dict[str, Any],
    list[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Path,
    Path,
    Path,
]:
    root = Path(candidate_dir)
    manifest_path = root / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing immutable base manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("candidate pool is not draft_kv_openhermes_stage2")
    records_path = root / str(manifest["records_file"])
    drafts_path = root / str(manifest["draft_cache_file"])
    if not records_path.is_file() or not drafts_path.is_file():
        raise RuntimeError("immutable base candidate pool is incomplete")
    if sha256_file(records_path) != str(manifest["records_sha256"]):
        raise RuntimeError("immutable base records SHA256 differs from manifest")
    records_by_id = read_jsonl_by_id(records_path)
    expected_ids = [
        str(example_id)
        for split in ("train", "gate_val", "reserve_test")
        for example_id in manifest["split_ids"][split]
    ]
    records = [records_by_id[example_id] for example_id in expected_ids]
    if list(records_by_id) != expected_ids:
        raise RuntimeError("immutable base records are not in manifest order")
    drafts = read_jsonl_by_id(drafts_path)
    if set(drafts) != set(expected_ids):
        raise RuntimeError("immutable base draft cache IDs do not match records")
    for record in records:
        validate_draft_row(drafts[str(record["example_id"])], record)
    if int(sum(int(value) for value in manifest["counts"].values())) != len(records):
        raise RuntimeError("immutable base record count differs from manifest")
    return manifest, records, drafts, manifest_path, records_path, drafts_path


def _record_candidates(
    *,
    data_path: str,
    data_split: str,
    stage1_manifest_path: str,
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    seed: int,
    min_gold_tokens: int,
    max_gold_tokens: int,
    max_receiver_length: int,
    max_sharer_prompt_tokens: int,
    max_sharer_length: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Reproduce the base deterministic qualifying-record traversal."""

    source = load_openhermes_records(data_path, split=data_split)
    excluded = _load_excluded_ids(stage1_manifest_path)
    order = list(range(len(source)))
    random.Random(int(seed)).shuffle(order)
    selected: list[Dict[str, Any]] = []
    seen_record_sha: set[str] = set()
    rejected: Counter[str] = Counter()
    for scanned, dataset_index in enumerate(order, start=1):
        example_id = f"openhermes:{dataset_index}"
        if example_id in excluded:
            rejected["excluded_stage1"] += 1
            continue
        try:
            row = build_stage2_record(source[dataset_index], dataset_index=dataset_index)
            if row["record_sha256"] in seen_record_sha:
                raise ValueError("duplicate normalized context and gold")
            receiver_full, _, receiver_target = receiver_target_encoding(
                receiver_tokenizer, row
            )
            sharer_prompt = sharer_prompt_token_ids(sharer_tokenizer, row)
            if not int(min_gold_tokens) <= len(receiver_target) <= int(max_gold_tokens):
                raise ValueError("gold token length")
            if len(receiver_full) > int(max_receiver_length):
                raise ValueError("Receiver sequence length")
            if len(sharer_prompt) > int(max_sharer_prompt_tokens):
                raise ValueError("Sharer prompt length")
            if len(sharer_prompt) + 1024 > int(max_sharer_length):
                raise ValueError("reserved Sharer sequence length")
        except (TypeError, ValueError) as error:
            rejected[str(error).split(":", 1)[0]] += 1
            continue
        seen_record_sha.add(str(row["record_sha256"]))
        row.update(
            {
                "receiver_prompt_tokens": int(
                    len(receiver_full) - len(receiver_target)
                ),
                "receiver_gold_tokens": len(receiver_target),
                "receiver_sequence_tokens": len(receiver_full),
                "sharer_prompt_tokens": len(sharer_prompt),
            }
        )
        selected.append(row)
    return selected, {
        "scanned_records": scanned,
        "qualified_records": len(selected),
        "excluded_stage1_count": len(excluded),
        "rejections": dict(sorted(rejected.items())),
    }


@torch.no_grad()
def _generate_rows(
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
        raise RuntimeError("extension cache contains unregistered candidate IDs")
    for example_id, row in existing.items():
        validate_draft_row(row, record_by_id[example_id], tokenizer)
    eos_value = tokenizer.eos_token_id
    eos_ids = (
        {int(value) for value in eos_value}
        if isinstance(eos_value, (tuple, list, set))
        else ({int(eos_value)} if eos_value is not None else set())
    )
    missing = [
        row for row in records if str(row["example_id"]) not in existing
    ]
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
                "record_sha256": str(record["record_sha256"]),
                "prompt_token_ids": prompt_rows[position],
                "draft_token_ids": draft_ids,
                "terminated_eos": bool(terminated),
                "response": response,
            }
            existing[str(row["example_id"])] = row
        ordered = [
            existing[str(row["example_id"])]
            for row in records
            if str(row["example_id"]) in existing
        ]
        write_jsonl(output_path, ordered)
        print(
            f"eos-pool extension drafts {len(existing)}/{len(records)}",
            flush=True,
        )
    return existing


def _assign_splits(
    records: Sequence[Mapping[str, Any]],
    drafts: Mapping[str, Mapping[str, Any]],
    *,
    train_count: int,
    gate_count: int,
    reserve_count: int,
    derangement_seed: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, list[str]]]:
    eos_rows = [
        dict(record)
        for record in records
        if bool(drafts[str(record["example_id"])]["terminated_eos"])
    ]
    required = int(train_count + gate_count + reserve_count)
    if len(eos_rows) < required:
        raise RuntimeError(
            f"EOS candidate pool has only {len(eos_rows)} rows; required {required}"
        )
    selected = eos_rows[:required]
    counts = {
        "train": int(train_count),
        "gate_val": int(gate_count),
        "reserve_test": int(reserve_count),
    }
    split_ids: Dict[str, list[str]] = {}
    split_drafts: Dict[str, Dict[str, Any]] = {}
    cursor = 0
    for split, count in counts.items():
        section = selected[cursor : cursor + count]
        for row in section:
            row["split"] = split
            example_id = str(row["example_id"])
            draft = dict(drafts[example_id])
            draft["split"] = split
            split_drafts[example_id] = draft
        split_ids[split] = [str(row["example_id"]) for row in section]
        cursor += count
    return selected, split_drafts, split_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draft-batch-size", type=int, default=32)
    parser.add_argument("--draft-max-new-tokens", type=int, default=1024)
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--gate-examples", type=int, default=512)
    parser.add_argument("--reserve-examples", type=int, default=512)
    parser.add_argument("--derangement-seed", type=int, default=20260829)
    args = parser.parse_args()

    seed_all(20260828)
    candidate = Path(args.candidate_dir)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"EOS-pool output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    (
        candidate_manifest,
        candidate_records,
        candidate_drafts,
        candidate_manifest_path,
        candidate_records_path,
        candidate_drafts_path,
    ) = _load_candidate_bundle(candidate)
    config = dict(candidate_manifest["command_config"])
    if config.get("require_eos") is not True:
        raise RuntimeError("base candidate command did not require EOS")
    if int(config["draft_max_new_tokens"]) != int(args.draft_max_new_tokens):
        raise RuntimeError("EOS-pool generation length differs from base candidate")
    # Generation batch is operational rather than semantic. Existing candidate
    # drafts are copied byte-for-byte, while extension rows may use a different
    # batch size; the final draft cache hash authenticates the exact mixture.
    draft_batch_sizes_used = {
        int(value)
        for value in candidate_manifest.get("draft_batch_sizes_used", [])
    }
    draft_batch_sizes_used.add(int(config["draft_batch_size"]))
    draft_batch_sizes_used.add(int(args.draft_batch_size))

    receiver_path = str(candidate_manifest["receiver"])
    sharer_path = str(candidate_manifest["sharer"])
    receiver_tokenizer = prepare_tokenizer(receiver_path, padding_side="right")
    sharer_tokenizer = prepare_tokenizer(sharer_path, padding_side="left")

    deterministic_candidates, selection_audit = _record_candidates(
        data_path=str(config["data"]),
        data_split=str(config["data_split"]),
        stage1_manifest_path=str(config["stage1_manifest"]),
        receiver_tokenizer=receiver_tokenizer,
        sharer_tokenizer=sharer_tokenizer,
        seed=int(config["seed"]),
        min_gold_tokens=int(config["min_gold_tokens"]),
        max_gold_tokens=int(config["max_gold_tokens"]),
        max_receiver_length=int(config["max_receiver_length"]),
        max_sharer_prompt_tokens=int(config["max_sharer_prompt_tokens"]),
        max_sharer_length=int(config["max_sharer_length"]),
    )
    candidate_pairs = [
        (str(row["example_id"]), str(row["record_sha256"]))
        for row in candidate_records
    ]
    if [
        (str(row["example_id"]), str(row["record_sha256"]))
        for row in deterministic_candidates[: len(candidate_records)]
    ] != candidate_pairs:
        raise RuntimeError("immutable base records are not the deterministic prefix")

    eos_existing = sum(
        bool(candidate_drafts[str(row["example_id"])]["terminated_eos"])
        for row in candidate_records
    )
    required = int(args.train_examples + args.gate_examples + args.reserve_examples)
    extra_needed = max(0, required - eos_existing)
    extension_records: list[Dict[str, Any]] = []
    extension_drafts: Dict[str, Dict[str, Any]] = {}
    if extra_needed:
        all_extension_candidates = deterministic_candidates[len(candidate_records) :]
        if not all_extension_candidates:
            raise RuntimeError("deterministic candidate traversal has no extension")
        sharer = load_causal_lm(sharer_path, torch.device(args.device))
        extension_cache_path = output / "candidate_extension_drafts.jsonl"
        cached_extension = read_jsonl_by_id(extension_cache_path)
        if cached_extension:
            cached_ids = set(cached_extension)
            expected_cached_ids = {
                str(row["example_id"])
                for row in all_extension_candidates[: len(cached_extension)]
            }
            if cached_ids != expected_cached_ids:
                raise RuntimeError(
                    "existing EOS-pool extension cache is not a deterministic prefix"
                )
        prefix_len = len(cached_extension)
        while True:
            prefix_len = min(
                len(all_extension_candidates),
                max(prefix_len + int(args.draft_batch_size), int(args.draft_batch_size)),
            )
            extension_records = all_extension_candidates[:prefix_len]
            extension_drafts = _generate_rows(
                model=sharer,
                tokenizer=sharer_tokenizer,
                records=extension_records,
                output_path=extension_cache_path,
                batch_size=args.draft_batch_size,
                max_new_tokens=args.draft_max_new_tokens,
            )
            eos_total = eos_existing + sum(
                bool(row["terminated_eos"]) for row in extension_drafts.values()
            )
            if eos_total >= required:
                break
            if prefix_len >= len(all_extension_candidates):
                raise RuntimeError("could not extend deterministic candidate pool")
            print(
                f"EOS pool still short: {eos_total}/{required}; "
                f"extending candidates to {prefix_len + int(args.draft_batch_size)}",
                flush=True,
            )

    all_records = candidate_records + extension_records
    all_drafts = dict(candidate_drafts) | dict(extension_drafts)
    selected, final_drafts, split_ids = _assign_splits(
        all_records,
        all_drafts,
        train_count=args.train_examples,
        gate_count=args.gate_examples,
        reserve_count=args.reserve_examples,
        derangement_seed=args.derangement_seed,
    )
    final_records_path = output / "stage2_records.jsonl"
    final_drafts_path = output / "sharer_drafts.jsonl"
    write_jsonl(final_records_path, selected)
    write_jsonl(
        final_drafts_path,
        [final_drafts[str(row["example_id"])] for row in selected],
    )
    selected_drafts = read_jsonl_by_id(final_drafts_path)
    final_record_ids = [str(row["example_id"]) for row in selected]
    manifest = {
        "protocol": EOS_POOL_PROTOCOL,
        "mode": "eos_pool",
        "candidate_pool_protocol": PROTOCOL,
        "candidate_pool_manifest": str(candidate_manifest_path.resolve()),
        "candidate_pool_manifest_sha256": sha256_file(candidate_manifest_path),
        "candidate_pool_records_sha256": sha256_file(candidate_records_path),
        "candidate_pool_drafts_sha256": sha256_file(candidate_drafts_path),
        "candidate_pool_count": len(all_records),
        "candidate_pool_eos_count": sum(
            bool(all_drafts[str(row["example_id"])]["terminated_eos"])
            for row in all_records
        ),
        "additional_candidate_count": len(extension_records),
        "receiver": receiver_path,
        "sharer": sharer_path,
        "layer_mapping": {
            str(target): int(source)
            for target, source in parse_layer_mapping(
                ",".join(
                    f"{target}:{source}"
                    for target, source in candidate_manifest["layer_mapping"].items()
                )
            ).items()
        },
        "stage1_manifest": str(config["stage1_manifest"]),
        "stage1_manifest_sha256": sha256_file(config["stage1_manifest"]),
        "source_data": str(config["data"]),
        "source_data_sha256": sha256_file(config["data"]),
        "source_data_split": str(config["data_split"]),
        "command_config": {
            **config,
            "protocol_mode": "eos_pool",
            "candidate_pool_manifest_sha256": sha256_file(candidate_manifest_path),
            "candidate_pool_filter": "terminated_eos == True",
            "candidate_pool_order": "immutable base order plus deterministic continuation",
            "eos_pool_train_examples": int(args.train_examples),
            "eos_pool_gate_examples": int(args.gate_examples),
            "eos_pool_reserve_examples": int(args.reserve_examples),
        },
        "draft_batch_sizes_used": sorted(draft_batch_sizes_used),
        "records_file": final_records_path.name,
        "records_sha256": sha256_file(final_records_path),
        "draft_cache_file": final_drafts_path.name,
        "text_control_header": TEXT_DRAFT_HEADER,
        "lengths": {
            "max_receiver_length": int(config["max_receiver_length"]),
            "max_receiver_text_length": int(config["max_receiver_text_length"]),
            "max_sharer_length": int(config["max_sharer_length"]),
        },
        "counts": {
            "train": int(args.train_examples),
            "gate_val": int(args.gate_examples),
            "reserve_test": int(args.reserve_examples),
        },
        "split_ids": split_ids,
        "gate_derangement": make_no_fixed_point_id_mapping(
            split_ids["gate_val"], seed=int(args.derangement_seed)
        ),
        "reserve_derangement": make_no_fixed_point_id_mapping(
            split_ids["reserve_test"], seed=int(args.derangement_seed) + 1
        ),
        "static_donor_id": split_ids["train"][0],
        "selection": {
            **selection_audit,
            "candidate_pool_count": len(all_records),
            "candidate_pool_eos_count": sum(
                bool(all_drafts[str(row["example_id"])]["terminated_eos"])
                for row in all_records
            ),
            "selected_count": len(selected),
            "filter": "terminated_eos == True",
            "performance_independent": True,
        },
    }
    manifest_path = output / "data_manifest.json"
    write_json(manifest_path, manifest)
    split_summary: Dict[str, Any] = {}
    for split in ("train", "gate_val", "reserve_test"):
        rows = [
            selected_drafts[str(row["example_id"])]
            for row in selected
            if row["split"] == split
        ]
        split_summary[split] = {
            "count": len(rows),
            "mean_draft_tokens": sum(len(row["draft_token_ids"]) for row in rows)
            / len(rows),
            "terminated_eos": sum(bool(row["terminated_eos"]) for row in rows),
        }
    result = {
        "protocol": EOS_POOL_PROTOCOL,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "records": str(final_records_path.resolve()),
        "records_sha256": sha256_file(final_records_path),
        "draft_cache": str(final_drafts_path.resolve()),
        "draft_cache_sha256": sha256_file(final_drafts_path),
        "splits": split_summary,
        "truncated_draft_count": 0,
        "candidate_pool_count": len(all_records),
        "candidate_pool_eos_count": manifest["candidate_pool_eos_count"],
        "additional_candidate_count": len(extension_records),
    }
    write_json(output / "prepare_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
