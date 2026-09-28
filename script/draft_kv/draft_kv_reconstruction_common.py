"""Shared training and evaluation helpers for Draft-KV text reconstruction."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from draft_kv.model.draft_kv import DraftKVPacket
from draft_kv.train.draft_kv_reconstruction_data import (
    PROTOCOL,
    RECONSTRUCTION_SYSTEM,
    RECONSTRUCTION_USER,
    extract_private_key,
    load_reconstruction_records,
    normalize_message_text,
    reconstruction_prompt_input_ids,
)
from script.draft_kv.draft_kv_common import bootstrap_ci, sha256_file


CONDITIONS = ("matched", "deranged", "static", "zero")
TRANSMISSION_SUFFIX = re.compile(
    r"\s*Transmission\s+key\s*:\s*DRAFT-KV-KEY-[0-9A-F]{12}",
    re.IGNORECASE,
)


def load_reconstruction_bundle(
    data_dir: str | Path,
) -> tuple[Dict[str, Any], list[Dict[str, Any]], Path, Path]:
    root = Path(data_dir)
    manifest_path = root / "data_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing reconstruction manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("unexpected reconstruction manifest protocol")
    records_path = root / str(manifest["records_file"])
    if sha256_file(records_path) != str(manifest["records_sha256"]):
        raise RuntimeError("reconstruction record SHA256 differs from manifest")
    records = load_reconstruction_records(records_path)
    split_ids = {
        str(split): [str(value) for value in values]
        for split, values in manifest["split_ids"].items()
    }
    expected_order = [
        example_id for values in split_ids.values() for example_id in values
    ]
    if len(expected_order) != len(set(expected_order)):
        raise RuntimeError("manifest split IDs are not globally unique")
    observed_ids = [str(row["example_id"]) for row in records]
    if expected_order != observed_ids:
        raise RuntimeError("manifest split IDs do not match reconstruction records")
    expected_split = {
        example_id: split
        for split, values in split_ids.items()
        for example_id in values
    }
    for row in records:
        example_id = str(row["example_id"])
        if str(row.get("split")) != expected_split[example_id]:
            raise RuntimeError(f"record split differs from manifest for {example_id}")
        message = str(row["message"])
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if digest != str(row["message_sha256"]):
            raise RuntimeError(f"message SHA256 mismatch for {example_id}")

    private_keys = [str(row.get("private_key", "")) for row in records]
    if bool(manifest.get("private_key_enabled")):
        if any(not key for key in private_keys) or len(private_keys) != len(
            set(private_keys)
        ):
            raise RuntimeError(
                "enabled private reconstruction keys are missing or reused"
            )
        if any(
            extract_private_key(str(row["message"])) != key.upper()
            for row, key in zip(records, private_keys)
        ):
            raise RuntimeError("private reconstruction key does not match its message")

    prompt = manifest.get("reconstruction_prompt", {})
    if prompt.get("system") != RECONSTRUCTION_SYSTEM or prompt.get(
        "user"
    ) != RECONSTRUCTION_USER:
        raise RuntimeError("manifest Receiver prompt differs from the protocol")

    train_ids = set(split_ids.get("train", []))
    static_id = str(manifest["static_donor_id"])
    if static_id not in train_ids:
        raise RuntimeError("static donor is not in the train split")
    for split, key in (
        ("gate_val", "gate_derangement"),
        ("reserve_test", "reserve_derangement"),
    ):
        ids = set(split_ids.get(split, []))
        mapping = {str(left): str(right) for left, right in manifest[key].items()}
        if (
            set(mapping) != ids
            or set(mapping.values()) != ids
            or any(left == right for left, right in mapping.items())
        ):
            raise RuntimeError(f"manifest {key} is not a full derangement")
    return manifest, records, manifest_path, records_path


def verify_receiver_prompt(
    tokenizer: Any, manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    """Re-encode and verify the sample-independent Receiver prompt contract."""

    token_ids = reconstruction_prompt_input_ids(tokenizer)
    digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected = manifest["reconstruction_prompt"]
    if len(token_ids) != int(expected["token_count"]):
        raise RuntimeError("Receiver prompt token count differs from manifest")
    if digest != str(expected["receiver_token_ids_sha256"]):
        raise RuntimeError("Receiver prompt token IDs differ from manifest")
    return {"token_count": len(token_ids), "token_ids_sha256": digest}


def validate_go_prerequisite(
    result_path: str | Path,
    *,
    expected_split: str,
    data_manifest_path: str | Path,
    records_path: str | Path,
    checkpoint_sha256: str | None = None,
) -> Dict[str, Any]:
    """Validate a prior evaluation before allowing the next protocol gate."""

    path = Path(result_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing prerequisite result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("protocol") != PROTOCOL:
        raise RuntimeError("prerequisite result has an unexpected protocol")
    if result.get("split") != str(expected_split):
        raise RuntimeError("prerequisite result is from the wrong split")
    if result.get("decision") != "GO":
        raise RuntimeError("prerequisite evaluation is not GO")

    eval_manifest_path = path.parent / "eval_manifest.json"
    if not eval_manifest_path.is_file():
        raise FileNotFoundError(
            f"missing prerequisite eval manifest: {eval_manifest_path}"
        )
    eval_manifest_sha = sha256_file(eval_manifest_path)
    if str(result.get("eval_manifest_sha256")) != eval_manifest_sha:
        raise RuntimeError("prerequisite eval manifest SHA256 differs from result")
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if eval_manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("prerequisite eval manifest has an unexpected protocol")
    if eval_manifest.get("split") != str(expected_split):
        raise RuntimeError("prerequisite eval manifest is from the wrong split")
    if str(eval_manifest.get("data_manifest_sha256")) != sha256_file(
        data_manifest_path
    ):
        raise RuntimeError("prerequisite uses a different data manifest")
    if str(eval_manifest.get("records_sha256")) != sha256_file(records_path):
        raise RuntimeError("prerequisite uses different reconstruction records")
    result_checkpoint_sha = str(result.get("checkpoint_sha256"))
    if result_checkpoint_sha != str(eval_manifest.get("checkpoint_sha256")):
        raise RuntimeError("prerequisite checkpoint SHA256 is internally inconsistent")
    if (
        checkpoint_sha256 is not None
        and result_checkpoint_sha != str(checkpoint_sha256)
    ):
        raise RuntimeError("prerequisite used a different checkpoint")
    return {
        "result": str(path.resolve()),
        "result_sha256": sha256_file(path),
        "eval_manifest": str(eval_manifest_path.resolve()),
        "eval_manifest_sha256": eval_manifest_sha,
        "checkpoint_sha256": result_checkpoint_sha,
        "decision": "GO",
    }


def token_reconstruction_stats(logits: Tensor, labels: Tensor) -> Dict[str, Tensor]:
    """Return per-example causal NLL and teacher-forced token accuracy."""

    labels = labels.to(logits.device)
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    losses = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted_labels.shape)
    valid = shifted_labels.ne(-100)
    counts = valid.sum(dim=1)
    if not bool(counts.gt(0).all()):
        raise ValueError("each reconstruction example must contain a target token")
    sums = (losses * valid).sum(dim=1)
    correct = (shifted_logits.argmax(dim=-1).eq(shifted_labels) & valid).sum(dim=1)
    return {
        "sum": sums,
        "count": counts,
        "mean": sums / counts.clamp_min(1),
        "correct": correct,
        "accuracy": correct / counts.clamp_min(1),
    }


@torch.no_grad()
def greedy_reconstruct(
    model: Any,
    tokenizer: Any,
    prompt_ids: Tensor,
    *,
    packet: Optional[DraftKVPacket],
    disable_communication: bool,
    max_new_tokens: int,
) -> str:
    """Greedily decode one message through the actual frozen Receiver LM head."""

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("greedy reconstruction currently requires batch size one")
    prompt_ids = prompt_ids.to(model.device)
    attention_mask = torch.ones_like(prompt_ids)
    logits, cache, _ = model.forward_receiver(
        prompt_ids,
        attention_mask,
        packet=packet,
        disable_communication=disable_communication,
        use_cache=True,
    )
    eos_values = set()
    for value in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(
            getattr(model.receiver, "generation_config", None),
            "eos_token_id",
            None,
        ),
    ):
        if isinstance(value, (tuple, list, set)):
            eos_values.update(int(item) for item in value)
        elif value is not None:
            eos_values.add(int(value))
    generated: list[int] = []
    next_token = logits[:, -1].argmax(dim=-1)
    for _ in range(int(max_new_tokens)):
        token = int(next_token.item())
        generated.append(token)
        if token in eos_values:
            break
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1
        )
        logits, cache, _ = model.forward_receiver(
            next_token.view(1, 1),
            attention_mask,
            packet=packet,
            disable_communication=disable_communication,
            past_key_values=cache,
            use_cache=True,
        )
        next_token = logits[:, -1].argmax(dim=-1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def normalized_exact(left: str, right: str) -> bool:
    return normalize_message_text(left) == normalize_message_text(right)


def sequence_match_ratio(left: str, right: str) -> float:
    return float(
        difflib.SequenceMatcher(
            None,
            normalize_message_text(left),
            normalize_message_text(right),
            autojunk=False,
        ).ratio()
    )


def natural_payload(text: str) -> str:
    """Remove the protocol canary suffix before scoring natural-message fidelity."""

    normalized = normalize_message_text(text)
    match = TRANSMISSION_SUFFIX.search(normalized)
    return normalize_message_text(normalized[: match.start()] if match else normalized)


def _condition_map(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    result: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        condition = str(row["condition"])
        example_id = str(row["example_id"])
        if condition not in CONDITIONS:
            raise ValueError(f"unknown reconstruction condition: {condition}")
        if example_id in result.setdefault(condition, {}):
            raise ValueError(f"duplicate {condition}/{example_id} evaluation row")
        result[condition][example_id] = row
    return result


def summarize_reconstruction(
    nll_rows: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Apply the preregistered channel-fidelity gates.

    Generation-side gates (private-key accuracy, donor following, sequence
    ratios) were removed on 2026-08-31: no Sharer configuration ever passed
    them and they do not reflect NLL reconstruction quality. Generation
    metrics are still computed and reported, but only the NLL-based
    statistical gates participate in the GO/NO_GO decision.
    """

    by_condition = _condition_map(nll_rows)
    expected = set(by_condition.get("matched", {}))
    if not expected or any(
        set(by_condition.get(name, {})) != expected for name in CONDITIONS
    ):
        raise ValueError("all NLL conditions must cover the same non-empty example set")
    if any(
        not np.isfinite(float(row["nll"])) or int(row["token_count"]) <= 0
        for row in nll_rows
    ):
        raise ValueError("NLL rows contain non-finite values or empty targets")
    means = {
        name: float(
            np.mean([float(by_condition[name][key]["nll"]) for key in sorted(expected)])
        )
        for name in CONDITIONS
    }
    token_accuracy = {
        name: float(
            sum(int(by_condition[name][key]["token_correct"]) for key in expected)
            / max(
                1,
                sum(
                    int(by_condition[name][key]["token_count"]) for key in expected
                ),
            )
        )
        for name in CONDITIONS
    }
    comparisons = {}
    for control in ("deranged", "static", "zero"):
        differences = [
            float(by_condition[control][key]["nll"])
            - float(by_condition["matched"][key]["nll"])
            for key in sorted(expected)
        ]
        comparisons[f"{control}_minus_matched"] = bootstrap_ci(
            differences,
            samples=int(bootstrap_samples),
            seed=int(seed),
        )
    relative_recovery = (
        (means["zero"] - means["matched"]) / means["zero"]
        if means["zero"] > 0
        else 0.0
    )

    matched_generation = [
        row for row in generation_rows if str(row.get("condition")) == "matched"
    ]
    deranged_generation = [
        row for row in generation_rows if str(row.get("condition")) == "deranged"
    ]
    generation_ids = {str(row["example_id"]) for row in matched_generation}
    if generation_ids != {str(row["example_id"]) for row in deranged_generation}:
        raise ValueError(
            "matched and deranged generation rows must cover identical IDs"
        )
    if not generation_ids.issubset(expected):
        raise ValueError("generation rows contain IDs outside the NLL evaluation")
    if len(matched_generation) != len(generation_ids) or len(
        deranged_generation
    ) != len(generation_ids):
        raise ValueError("generation rows contain duplicate condition/example IDs")
    generation_count = len(generation_ids)

    def fraction(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0

    generation = {
        "count": generation_count,
        "matched_private_key_accuracy": fraction(
            matched_generation, "private_key_matches_donor"
        ),
        "deranged_donor_private_key_accuracy": fraction(
            deranged_generation, "private_key_matches_donor"
        ),
        "donor_following_fraction": fraction(deranged_generation, "follows_donor"),
        "matched_full_exact_match": fraction(matched_generation, "exact_to_donor"),
        "deranged_full_exact_to_donor": fraction(
            deranged_generation, "exact_to_donor"
        ),
        "matched_mean_sequence_ratio": float(
            np.mean(
                [float(row["sequence_ratio_to_donor"]) for row in matched_generation]
            )
        )
        if matched_generation
        else 0.0,
        "matched_mean_natural_sequence_ratio": float(
            np.mean(
                [
                    float(row["natural_sequence_ratio_to_donor"])
                    for row in matched_generation
                ]
            )
        )
        if matched_generation
        else 0.0,
        "deranged_mean_natural_sequence_ratio_to_donor": float(
            np.mean(
                [
                    float(row["natural_sequence_ratio_to_donor"])
                    for row in deranged_generation
                ]
            )
        )
        if deranged_generation
        else 0.0,
        "deranged_mean_sequence_ratio_to_donor": float(
            np.mean(
                [float(row["sequence_ratio_to_donor"]) for row in deranged_generation]
            )
        )
        if deranged_generation
        else 0.0,
    }
    statistical_pass = all(
        comparisons[f"{control}_minus_matched"]["ci95_lower"] > 0.0
        for control in ("deranged", "static", "zero")
    )
    matched_fraction_pass = (
        comparisons["deranged_minus_matched"]["positive_fraction"] > 0.60
    )
    recovery_pass = relative_recovery >= 0.20
    if generation_count == 0:
        decision = "INCOMPLETE_NO_GENERATION"
    elif statistical_pass and matched_fraction_pass and recovery_pass:
        decision = "GO"
    else:
        decision = "NO_GO"
    return {
        "decision": decision,
        "count": len(expected),
        "example_mean_nll": means,
        "token_accuracy": token_accuracy,
        "comparisons": comparisons,
        "relative_recovery_from_zero": float(relative_recovery),
        "generation": generation,
        "gates": {
            "all_control_ci_lowers_positive": statistical_pass,
            "deranged_matched_better_fraction_gt_0_60": matched_fraction_pass,
            "relative_recovery_ge_0_20": recovery_pass,
        },
    }


def make_generation_row(
    *,
    condition: str,
    target: Mapping[str, Any],
    donor: Mapping[str, Any],
    response: str,
) -> Dict[str, Any]:
    extracted = extract_private_key(response)
    target_key = str(target.get("private_key", "")).upper() or None
    donor_key = str(donor.get("private_key", "")).upper() or None
    matches_donor = extracted is not None and extracted == donor_key
    response_natural = natural_payload(response)
    return {
        "condition": str(condition),
        "example_id": str(target["example_id"]),
        "donor_example_id": str(donor["example_id"]),
        "target_private_key": target_key,
        "donor_private_key": donor_key,
        "extracted_private_key": extracted,
        "private_key_matches_donor": bool(matches_donor),
        "follows_donor": bool(matches_donor and donor_key != target_key),
        "exact_to_target": normalized_exact(response, str(target["message"])),
        "exact_to_donor": normalized_exact(response, str(donor["message"])),
        "sequence_ratio_to_target": sequence_match_ratio(
            response, str(target["message"])
        ),
        "sequence_ratio_to_donor": sequence_match_ratio(
            response, str(donor["message"])
        ),
        "natural_exact_to_target": normalized_exact(
            response_natural, str(target["natural_message"])
        ),
        "natural_exact_to_donor": normalized_exact(
            response_natural, str(donor["natural_message"])
        ),
        "natural_sequence_ratio_to_target": sequence_match_ratio(
            response_natural, str(target["natural_message"])
        ),
        "natural_sequence_ratio_to_donor": sequence_match_ratio(
            response_natural, str(donor["natural_message"])
        ),
        "response": str(response),
    }


def read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(output)


__all__ = [
    "CONDITIONS",
    "greedy_reconstruct",
    "load_reconstruction_bundle",
    "make_generation_row",
    "natural_payload",
    "normalized_exact",
    "read_jsonl",
    "sequence_match_ratio",
    "summarize_reconstruction",
    "token_reconstruction_stats",
    "validate_go_prerequisite",
    "verify_receiver_prompt",
    "write_jsonl",
]
