"""Evaluate Stage-2 response-KV communication and beyond-text controls."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.model.draft_kv import DraftKVPacket  # noqa: E402
from draft_kv.train.draft_kv_openhermes_stage2_data import (  # noqa: E402
    DraftKVStage2Collator,
    PROTOCOL,
    SUPPORTED_PROTOCOLS,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    bootstrap_ci,
    build_model,
    load_trainable_state,
    seed_all,
    sha256_file,
    write_json,
)
from script.draft_kv.draft_kv_openhermes_stage2_common import (  # noqa: E402
    condition_nll_rows,
    load_stage2_bundle,
    make_stage2_dataset,
    summarize_condition_rows,
)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _paired_difference(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, float]:
    by_condition: Dict[str, Dict[str, float]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), {})[
            str(row["example_id"])
        ] = float(row["nll"])
    if left not in by_condition or right not in by_condition:
        raise RuntimeError(f"missing paired NLL condition: {left} or {right}")
    if set(by_condition[left]) != set(by_condition[right]):
        raise RuntimeError(f"paired NLL IDs differ for {left} and {right}")
    differences = [
        by_condition[left][example_id] - by_condition[right][example_id]
        for example_id in sorted(by_condition[left])
    ]
    return bootstrap_ci(differences, samples=bootstrap_samples, seed=seed)


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Mapping[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    comparisons = {
        "zero_minus_matched": ("zero", "matched"),
        "deranged_minus_matched": ("deranged", "matched"),
        "static_minus_matched": ("static", "matched"),
        "text_only_minus_text_matched": ("text_only", "text_matched"),
        "text_deranged_minus_text_matched": ("text_deranged", "text_matched"),
        "text_static_minus_text_matched": ("text_static", "text_matched"),
        "text_only_minus_matched": ("text_only", "matched"),
    }
    paired = {
        name: _paired_difference(
            rows,
            left,
            right,
            bootstrap_samples=bootstrap_samples,
            seed=int(seed) + offset,
        )
        for offset, (name, (left, right)) in enumerate(comparisons.items())
    }
    preservation = checkpoint.get("reconstruction_preservation", {})
    core = {
        "checkpoint_preserves_stage1_reconstruction": bool(
            preservation.get("eligible")
        ),
        "matched_beats_zero": paired["zero_minus_matched"]["ci95_lower"] > 0,
        "matched_beats_deranged": paired["deranged_minus_matched"]["ci95_lower"] > 0,
        "matched_beats_static": paired["static_minus_matched"]["ci95_lower"] > 0,
    }
    beyond_text = {
        "text_matched_beats_text_only": paired[
            "text_only_minus_text_matched"
        ]["ci95_lower"]
        > 0,
        "text_matched_beats_text_deranged": paired[
            "text_deranged_minus_text_matched"
        ]["ci95_lower"]
        > 0,
        "text_matched_beats_text_static": paired[
            "text_static_minus_text_matched"
        ]["ci95_lower"]
        > 0,
    }
    return {
        "condition_metrics": summarize_condition_rows(rows),
        "paired_bootstrap": paired,
        "stage2_core_criteria": core,
        "beyond_text_criteria": beyond_text,
        "decision": "GO" if all(core.values()) else "NO_GO",
        "beyond_text_evidence": "YES" if all(beyond_text.values()) else "NO",
    }


@torch.no_grad()
def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    packet: Optional[DraftKVPacket],
    disable_communication: bool,
    max_new_tokens: int,
) -> str:
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=model.device)
    attention = torch.ones_like(ids)
    logits, cache, _ = model.forward_receiver(
        ids,
        attention,
        packet=packet,
        disable_communication=disable_communication,
        use_cache=True,
    )
    eos_value = tokenizer.eos_token_id
    eos_ids = (
        {int(value) for value in eos_value}
        if isinstance(eos_value, (tuple, list, set))
        else ({int(eos_value)} if eos_value is not None else set())
    )
    generated: list[int] = []
    next_token = logits[:, -1].argmax(dim=-1)
    for _ in range(int(max_new_tokens)):
        value = int(next_token.item())
        generated.append(value)
        if value in eos_ids:
            break
        attention = torch.cat(
            [attention, torch.ones((1, 1), dtype=attention.dtype, device=model.device)],
            dim=1,
        )
        logits, cache, _ = model.forward_receiver(
            next_token.view(1, 1),
            attention,
            packet=packet,
            disable_communication=disable_communication,
            past_key_values=cache,
            use_cache=True,
        )
        next_token = logits[:, -1].argmax(dim=-1)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _packet_for(
    model: Any,
    collator: DraftKVStage2Collator,
    row: Mapping[str, Any],
) -> DraftKVPacket:
    batch = collator([row])
    return model.make_packet(
        batch["sharer_input_ids"],
        batch["sharer_attention_mask"],
        batch["sharer_draft_mask"],
        detach=True,
    )


@torch.no_grad()
def generation_rows(
    model: Any,
    tokenizer: Any,
    context_dataset: Any,
    text_dataset: Any,
    packet_rows: Mapping[str, Mapping[str, Any]],
    collator: DraftKVStage2Collator,
    *,
    derangement: Mapping[str, str],
    count: int,
    max_new_tokens: int,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    selected = context_dataset.rows[: int(count)]
    for position, context_row in enumerate(selected, start=1):
        example_id = str(context_row["example_id"])
        text_row = text_dataset.by_id[example_id]
        donor_id = str(derangement[example_id])
        matched_packet = _packet_for(model, collator, packet_rows[example_id])
        deranged_packet = _packet_for(model, collator, packet_rows[donor_id])
        arms = (
            ("zero", context_row, None, True, None),
            ("matched", context_row, matched_packet, False, example_id),
            ("deranged", context_row, deranged_packet, False, donor_id),
            ("text_only", text_row, None, True, None),
            ("text_matched", text_row, matched_packet, False, example_id),
            ("text_deranged", text_row, deranged_packet, False, donor_id),
        )
        gold = str(context_row["gold_message"]).strip()
        for condition, receiver_row, packet, disable, packet_donor in arms:
            response = greedy_generate(
                model,
                tokenizer,
                receiver_row["receiver_prompt_input_ids"],
                packet=packet,
                disable_communication=disable,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    "condition": condition,
                    "example_id": example_id,
                    "packet_donor_example_id": packet_donor,
                    "gold": gold,
                    "response": response,
                    "exact_match": response == gold,
                    "sequence_ratio": difflib.SequenceMatcher(None, response, gold).ratio(),
                }
            )
        print(f"generation {position}/{len(selected)}", flush=True)
    return rows


def _validate_gate_prerequisite(
    path: str | None,
    *,
    protocol: str,
    checkpoint_sha: str,
    manifest_sha: str,
) -> None:
    if not path:
        raise RuntimeError("reserve_test requires --require-gate-go")
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        result.get("protocol") != protocol
        or result.get("split") != "gate_val"
        or result.get("decision") != "GO"
        or result.get("checkpoint_sha256") != checkpoint_sha
        or result.get("data_manifest_sha256") != manifest_sha
    ):
        raise RuntimeError("gate prerequisite is not a matching Stage-2 GO")
    gate_manifest_path = Path(path).parent / "eval_manifest.json"
    if not gate_manifest_path.is_file():
        raise RuntimeError("gate prerequisite has no eval_manifest.json")
    if result.get("eval_manifest_sha256") != sha256_file(gate_manifest_path):
        raise RuntimeError("gate prerequisite eval manifest SHA mismatch")
    gate_manifest = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    if (
        gate_manifest.get("protocol") != protocol
        or gate_manifest.get("split") != "gate_val"
        or gate_manifest.get("checkpoint_sha256") != checkpoint_sha
        or gate_manifest.get("data_manifest_sha256") != manifest_sha
    ):
        raise RuntimeError("gate eval manifest is not a matching prerequisite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("gate_val", "reserve_test"), default="gate_val")
    parser.add_argument("--require-gate-go")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--generation-examples", type=int, default=64)
    parser.add_argument("--generation-max-new-tokens", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.eval_batch_size, args.generation_max_new_tokens, args.bootstrap_samples) <= 0:
        raise ValueError("evaluation batch, generation, and bootstrap sizes must be positive")
    if args.generation_examples < 0:
        raise ValueError("generation_examples cannot be negative")
    seed_all(args.seed)
    output = Path(args.output_dir)
    artifacts = (
        output / "eval_manifest.json",
        output / "eval_result.json",
        output / "per_example_nll.jsonl",
        output / "per_example_generation.jsonl",
    )
    if any(path.exists() for path in artifacts):
        raise RuntimeError("Stage 2 evaluation output exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)

    manifest, records, drafts, manifest_path, records_path, drafts_path = (
        load_stage2_bundle(args.data_dir)
    )
    if int(manifest["command_config"]["seed"]) != int(args.seed):
        raise RuntimeError("Stage-2 evaluation seed differs from data manifest")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol") != PROTOCOL:
        if checkpoint.get("protocol") not in SUPPORTED_PROTOCOLS:
            raise RuntimeError("evaluation checkpoint has an unexpected protocol")
    protocol = str(manifest["protocol"])
    if checkpoint.get("protocol") != protocol:
        raise RuntimeError("evaluation checkpoint protocol differs from data manifest")
    if checkpoint.get("stage2_data_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("checkpoint was trained on a different Stage 2 manifest")
    if not bool(checkpoint.get("reconstruction_preservation", {}).get("eligible")):
        raise RuntimeError("checkpoint did not preserve Stage-1 reconstruction")
    if args.split == "reserve_test":
        _validate_gate_prerequisite(
            args.require_gate_go,
            protocol=protocol,
            checkpoint_sha=checkpoint_sha,
            manifest_sha=sha256_file(manifest_path),
        )
    elif args.require_gate_go:
        raise RuntimeError("gate_val evaluation does not accept --require-gate-go")

    model, receiver_tokenizer, sharer_tokenizer = build_model(
        receiver_path=str(manifest["receiver"]),
        sharer_path=str(manifest["sharer"]),
        layer_mapping=manifest["layer_mapping"],
        device_name=args.device,
    )
    load_trainable_state(model, checkpoint)
    model.set_stage("eval")
    collator = DraftKVStage2Collator(receiver_tokenizer, sharer_tokenizer)
    context_dataset = make_stage2_dataset(
        manifest,
        records,
        drafts,
        receiver_tokenizer,
        sharer_tokenizer,
        split=args.split,
        receiver_mode="context",
    )
    text_dataset = make_stage2_dataset(
        manifest,
        records,
        drafts,
        receiver_tokenizer,
        sharer_tokenizer,
        split=args.split,
        receiver_mode="text",
    )
    train_dataset = make_stage2_dataset(
        manifest,
        records,
        drafts,
        receiver_tokenizer,
        sharer_tokenizer,
        split="train",
        receiver_mode="context",
    )
    packet_rows = dict(train_dataset.by_id) | dict(context_dataset.by_id)
    derangement = manifest[
        "gate_derangement" if args.split == "gate_val" else "reserve_derangement"
    ]
    context_rows = condition_nll_rows(
        model,
        context_dataset,
        packet_rows,
        collator,
        derangement=derangement,
        static_donor_id=str(manifest["static_donor_id"]),
        batch_size=args.eval_batch_size,
    )
    text_rows = condition_nll_rows(
        model,
        text_dataset,
        packet_rows,
        collator,
        derangement=derangement,
        static_donor_id=str(manifest["static_donor_id"]),
        batch_size=args.eval_batch_size,
        condition_prefix="text_",
    )
    for row in text_rows:
        if row["condition"] == "text_zero":
            row["condition"] = "text_only"
    nll_rows = context_rows + text_rows
    result = summarize(
        nll_rows,
        checkpoint=checkpoint,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 1000,
    )
    generations = generation_rows(
        model,
        receiver_tokenizer,
        context_dataset,
        text_dataset,
        packet_rows,
        collator,
        derangement=derangement,
        count=min(int(args.generation_examples), len(context_dataset)),
        max_new_tokens=args.generation_max_new_tokens,
    ) if args.generation_examples else []
    generation_summary: Dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in generations}):
        values = [row for row in generations if row["condition"] == condition]
        generation_summary[condition] = {
            "count": len(values),
            "exact_match": sum(bool(row["exact_match"]) for row in values) / len(values),
            "mean_sequence_ratio": sum(float(row["sequence_ratio"]) for row in values) / len(values),
        }
    eval_manifest = {
        "protocol": protocol,
        "split": args.split,
        "data_manifest_sha256": sha256_file(manifest_path),
        "records_sha256": sha256_file(records_path),
        "draft_cache_sha256": sha256_file(drafts_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "conditions": [
            "zero",
            "matched",
            "deranged",
            "static",
            "text_only",
            "text_matched",
            "text_deranged",
            "text_static",
        ],
        "generation_examples": int(args.generation_examples),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
    }
    write_json(output / "eval_manifest.json", eval_manifest)
    _write_jsonl(output / "per_example_nll.jsonl", nll_rows)
    _write_jsonl(output / "per_example_generation.jsonl", generations)
    result.update(
        {
            "protocol": protocol,
            "split": args.split,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "data_manifest_sha256": sha256_file(manifest_path),
            "eval_manifest_sha256": sha256_file(output / "eval_manifest.json"),
            "generation_metrics": generation_summary,
            "reconstruction_preservation": checkpoint["reconstruction_preservation"],
        }
    )
    write_json(output / "eval_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
