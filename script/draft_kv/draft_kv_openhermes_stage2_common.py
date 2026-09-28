"""Shared loading, validation, and NLL controls for OpenHermes Stage 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from draft_kv.train.draft_kv_openhermes_stage2_data import (
    DraftKVStage2Collator,
    SUPPORTED_PROTOCOLS,
    OpenHermesDraftKVStage2Dataset,
    read_jsonl_by_id,
)
from script.draft_kv.draft_kv_common import answer_nll, sha256_file


def load_stage2_bundle(
    data_dir: str | Path,
) -> tuple[Dict[str, Any], list[Dict[str, Any]], Dict[str, Dict[str, Any]], Path, Path, Path]:
    root = Path(data_dir)
    manifest_path = root / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = str(manifest.get("protocol", ""))
    if protocol not in SUPPORTED_PROTOCOLS:
        raise RuntimeError("unexpected Stage 2 data protocol")
    records_path = root / str(manifest["records_file"])
    drafts_path = root / str(manifest["draft_cache_file"])
    if sha256_file(records_path) != manifest.get("records_sha256"):
        raise RuntimeError("Stage 2 records SHA mismatch")
    records = list(read_jsonl_by_id(records_path).values())
    drafts = read_jsonl_by_id(drafts_path)
    expected_ids = [str(row["example_id"]) for row in records]
    registered_ids = [
        str(example_id)
        for split in ("train", "gate_val", "reserve_test")
        for example_id in manifest["split_ids"][split]
    ]
    if expected_ids != registered_ids or len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("Stage 2 record order/IDs differ from manifest")
    expected_split = {
        str(example_id): split
        for split, values in manifest["split_ids"].items()
        for example_id in values
    }
    if any(str(row.get("split")) != expected_split[str(row["example_id"])] for row in records):
        raise RuntimeError("Stage 2 record split differs from manifest")
    if set(expected_ids) != set(drafts):
        raise RuntimeError("Stage 2 draft cache IDs do not equal record IDs")
    if any(not bool(drafts[example_id].get("terminated_eos")) for example_id in expected_ids):
        raise RuntimeError("Stage 2 bundle contains a truncated Sharer response")
    result_path = root / "prepare_result.json"
    if not result_path.is_file():
        raise RuntimeError("Stage 2 bundle has no completed prepare_result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("protocol") != protocol
        or result.get("manifest_sha256") != sha256_file(manifest_path)
        or result.get("records_sha256") != sha256_file(records_path)
        or result.get("draft_cache_sha256") != sha256_file(drafts_path)
        or int(result.get("truncated_draft_count", -1)) != 0
    ):
        raise RuntimeError("Stage 2 prepare result does not authenticate the bundle")
    for split, expected_count in manifest["counts"].items():
        observed = sum(str(row.get("split")) == str(split) for row in records)
        if observed != int(expected_count):
            raise RuntimeError(f"Stage 2 {split} count differs from manifest")
    train_ids = set(str(value) for value in manifest["split_ids"]["train"])
    if str(manifest["static_donor_id"]) not in train_ids:
        raise RuntimeError("Stage 2 static donor is not in train")
    for split, name in (
        ("gate_val", "gate_derangement"),
        ("reserve_test", "reserve_derangement"),
    ):
        ids = set(str(value) for value in manifest["split_ids"][split])
        mapping = {
            str(target): str(donor)
            for target, donor in manifest[name].items()
        }
        if (
            set(mapping) != ids
            or set(mapping.values()) != ids
            or any(target == donor for target, donor in mapping.items())
        ):
            raise RuntimeError(f"Stage 2 {name} is not a full derangement")
    return manifest, records, drafts, manifest_path, records_path, drafts_path


def make_stage2_dataset(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    drafts: Mapping[str, Mapping[str, Any]],
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    *,
    split: str,
    receiver_mode: str = "context",
) -> OpenHermesDraftKVStage2Dataset:
    return OpenHermesDraftKVStage2Dataset(
        records,
        drafts,
        receiver_tokenizer,
        sharer_tokenizer,
        split=split,
        receiver_mode=receiver_mode,
        max_receiver_length=int(
            manifest["lengths"][
                "max_receiver_text_length"
                if receiver_mode == "text"
                else "max_receiver_length"
            ]
        ),
        max_sharer_length=int(manifest["lengths"]["max_sharer_length"]),
    )


@torch.no_grad()
def parity_checks(model: Any, batch: Mapping[str, Any]) -> Dict[str, float]:
    """Check native/split and true zero-gate identities after Stage-1 init."""

    model.set_stage("eval")
    ids = batch["receiver_input_ids"].to(model.device)
    mask = batch["receiver_attention_mask"].to(model.device)
    native = model.receiver(input_ids=ids, attention_mask=mask, use_cache=False).logits
    base, _, _ = model.forward_receiver(
        ids,
        mask,
        packet=None,
        disable_communication=True,
        use_cache=False,
    )
    saved = {
        key: module.gate_logits.detach().clone()
        for key, module in model.consumer.external.items()
    }
    try:
        for module in model.consumer.external.values():
            module.gate_logits.zero_()
        packet = model.make_packet(
            batch["sharer_input_ids"],
            batch["sharer_attention_mask"],
            batch["sharer_draft_mask"],
            detach=True,
        )
        zero_gate, _, _ = model.forward_receiver(
            ids,
            mask,
            packet=packet,
            disable_communication=False,
            use_cache=False,
        )
    finally:
        for key, values in saved.items():
            model.consumer.external[key].gate_logits.copy_(values)
    return {
        "native_vs_split_max_abs": float((native - base).abs().max()),
        "base_vs_zero_gate_max_abs": float((base - zero_gate).abs().max()),
    }


def _donor_features(
    target_ids: Sequence[str],
    *,
    condition: str,
    packet_rows: Mapping[str, Mapping[str, Any]],
    derangement: Mapping[str, str],
    static_donor_id: str,
) -> list[Mapping[str, Any]]:
    if condition == "matched":
        donor_ids = list(target_ids)
    elif condition == "deranged":
        donor_ids = [str(derangement[target]) for target in target_ids]
    elif condition == "static":
        donor_ids = [str(static_donor_id)] * len(target_ids)
    else:
        raise ValueError(f"condition {condition!r} has no donor packet")
    return [packet_rows[donor_id] for donor_id in donor_ids]


@torch.no_grad()
def condition_nll_rows(
    model: Any,
    target_dataset: OpenHermesDraftKVStage2Dataset,
    packet_rows: Mapping[str, Mapping[str, Any]],
    collator: DraftKVStage2Collator,
    *,
    derangement: Mapping[str, str],
    static_donor_id: str,
    batch_size: int,
    condition_prefix: str = "",
) -> list[Dict[str, Any]]:
    """Evaluate Zero/Matched/Deranged/Static with paired target ordering."""

    model.set_stage("eval")
    loader = DataLoader(
        target_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    rows: list[Dict[str, Any]] = []
    for target_batch in loader:
        target_ids = [str(value) for value in target_batch["example_id"]]
        for condition in ("zero", "matched", "deranged", "static"):
            if condition == "zero":
                packet = None
                donor_ids = [None] * len(target_ids)
            else:
                donor_features = _donor_features(
                    target_ids,
                    condition=condition,
                    packet_rows=packet_rows,
                    derangement=derangement,
                    static_donor_id=static_donor_id,
                )
                donor_batch = collator(donor_features)
                packet = model.make_packet(
                    donor_batch["sharer_input_ids"],
                    donor_batch["sharer_attention_mask"],
                    donor_batch["sharer_draft_mask"],
                    detach=True,
                )
                donor_ids = [str(row["example_id"]) for row in donor_features]
            logits, _, _ = model.forward_receiver(
                target_batch["receiver_input_ids"],
                target_batch["receiver_attention_mask"],
                packet=packet,
                disable_communication=condition == "zero",
                use_cache=False,
            )
            stats = answer_nll(logits, target_batch["labels"])
            predictions = logits[:, :-1].argmax(dim=-1)
            labels = target_batch["labels"][:, 1:].to(logits.device)
            valid = labels.ne(-100)
            correct = (predictions.eq(labels) & valid).sum(dim=1)
            for index, example_id in enumerate(target_ids):
                rows.append(
                    {
                        "condition": condition_prefix + condition,
                        "example_id": example_id,
                        "donor_example_id": donor_ids[index],
                        "nll": float(stats["mean"][index]),
                        "token_correct": int(correct[index]),
                        "token_count": int(stats["count"][index]),
                    }
                )
    return rows


def summarize_condition_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_condition: Dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    result: Dict[str, Any] = {}
    for condition, values in sorted(by_condition.items()):
        result[condition] = {
            "count": len(values),
            "mean_nll": float(np.mean([float(row["nll"]) for row in values])),
            "token_accuracy": sum(int(row["token_correct"]) for row in values)
            / max(1, sum(int(row["token_count"]) for row in values)),
        }
    return result


__all__ = [
    "condition_nll_rows",
    "load_stage2_bundle",
    "make_stage2_dataset",
    "parity_checks",
    "summarize_condition_rows",
]
