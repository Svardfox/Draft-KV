"""Evaluate Draft-KV reconstruction under matched and causal controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_reconstruction_data import (  # noqa: E402
    DraftKVReconstructionCollator,
    OpenHermesDraftKVReconstructionDataset,
    PROTOCOL,
    make_no_fixed_point_id_mapping,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    build_model,
    load_trainable_state,
    seed_all,
    sha256_file,
    write_json,
)
from script.draft_kv.draft_kv_reconstruction_common import (  # noqa: E402
    CONDITIONS,
    greedy_reconstruct,
    load_reconstruction_bundle,
    make_generation_row,
    read_jsonl,
    summarize_reconstruction,
    token_reconstruction_stats,
    validate_go_prerequisite,
    verify_receiver_prompt,
    write_jsonl,
)


def _make_dataset(
    records: Sequence[Mapping[str, Any]],
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    manifest: Mapping[str, Any],
    *,
    split: str,
) -> OpenHermesDraftKVReconstructionDataset:
    lengths = manifest["lengths"]
    return OpenHermesDraftKVReconstructionDataset(
        records,
        receiver_tokenizer,
        sharer_tokenizer,
        split=split,
        min_message_tokens=int(lengths["min_message_tokens"]),
        max_message_tokens=int(lengths["max_message_tokens"]),
        max_receiver_length=int(lengths["max_receiver_length"]),
        max_sharer_length=int(lengths["max_sharer_length"]),
    )


@torch.no_grad()
def _native_split_parity(model: Any, batch: Mapping[str, Any]) -> float:
    ids = batch["receiver_input_ids"].to(model.device)
    mask = batch["receiver_attention_mask"].to(model.device)
    native = model.receiver(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
    ).logits
    split, _, _ = model.forward_receiver(
        ids,
        mask,
        packet=None,
        disable_communication=True,
        use_cache=False,
    )
    return float((native - split).abs().max())


def _source_packet(model: Any, collator: Any, rows: Sequence[Mapping[str, Any]]):
    batch = collator(rows)
    return model.make_packet(
        batch["sharer_input_ids"],
        batch["sharer_attention_mask"],
        batch["sharer_draft_mask"],
        detach=True,
    )


@torch.no_grad()
def evaluate_nll(
    model: Any,
    dataset: OpenHermesDraftKVReconstructionDataset,
    collator: DraftKVReconstructionCollator,
    donor_mapping: Mapping[str, str],
    static_row: Mapping[str, Any],
    *,
    batch_size: int,
) -> list[Dict[str, Any]]:
    model.set_stage("eval")
    rows: list[Dict[str, Any]] = []
    for start in range(0, len(dataset), int(batch_size)):
        targets = [
            dataset[index]
            for index in range(start, min(start + int(batch_size), len(dataset)))
        ]
        donors = [
            dataset.by_id[str(donor_mapping[row["example_id"]])] for row in targets
        ]
        static_rows = [static_row for _ in targets]
        target_batch = collator(targets)
        packets = {
            "matched": _source_packet(model, collator, targets),
            "deranged": _source_packet(model, collator, donors),
            "static": _source_packet(model, collator, static_rows),
            "zero": None,
        }
        for condition in CONDITIONS:
            logits, _, _ = model.forward_receiver(
                target_batch["receiver_input_ids"],
                target_batch["receiver_attention_mask"],
                packet=packets[condition],
                disable_communication=condition == "zero",
                use_cache=False,
            )
            stats = token_reconstruction_stats(logits, target_batch["labels"])
            for position, target in enumerate(targets):
                donor = (
                    donors[position]
                    if condition == "deranged"
                    else static_row
                    if condition == "static"
                    else target
                )
                rows.append(
                    {
                        "condition": condition,
                        "example_id": str(target["example_id"]),
                        "donor_example_id": (
                            None if condition == "zero" else str(donor["example_id"])
                        ),
                        "nll": float(stats["mean"][position]),
                        "nll_sum": float(stats["sum"][position]),
                        "token_count": int(stats["count"][position]),
                        "token_correct": int(stats["correct"][position]),
                    }
                )
        print(
            f"nll {min(start + len(targets), len(dataset))}/{len(dataset)}",
            flush=True,
        )
    return rows


def _generation_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["condition"]), str(row["example_id"])


def _validate_training_protocol(
    checkpoint: Mapping[str, Any],
    *,
    split: str,
    reconstruction_mode: str,
    expected_seed: int,
) -> None:
    """Reject non-final or off-protocol checkpoints from formal evaluation."""

    mode = "overfit" if split == "overfit" else "pilot"
    mode = str(reconstruction_mode or "base")
    if mode == "base":
        pilot_max_updates = 200
        pilot_eval_every = 50
        pilot_log_every = 10
    elif mode == "longrun":
        pilot_max_updates = 16000
        pilot_eval_every = 1000
        pilot_log_every = 100
    else:
        raise RuntimeError(f"unknown reconstruction mode: {mode}")
    expected = {
        "mode": mode,
        "projector_lr": 1e-3,
        "gate_lr": 1e-2,
        "eval_every": 50 if mode == "overfit" else pilot_eval_every,
        "log_every": 10 if mode == "overfit" else pilot_log_every,
        "seed": int(expected_seed),
    }
    # The multi-Sharer launcher uses a single-GPU conservative profile so the
    # 4B Sharer fits without changing the effective optimizer batch.  Accept
    # only these two fully specified profiles; do not treat arbitrary
    # microbatch/accumulation combinations as formal protocol runs.
    batch_profiles = {
        "standard": {
            "microbatch": 8 if mode == "overfit" else 16,
            "grad_accum": 4,
            "eval_batch_size": 16,
        },
        "single_gpu_conservative": {
            "microbatch": 1,
            "grad_accum": 32 if mode == "overfit" else 64,
            "eval_batch_size": 1,
        },
        "single_gpu_1p5b": {
            "microbatch": 4,
            "grad_accum": 8 if mode == "overfit" else 16,
            "eval_batch_size": 1,
        },
        "single_gpu_qwen3_8b": {
            "microbatch": 4,
            "grad_accum": 8 if mode == "overfit" else 16,
            "eval_batch_size": 1,
        },
        "single_gpu_1p5b_pilot_8": {
            "microbatch": 8,
            "grad_accum": 4 if mode == "overfit" else 8,
            "eval_batch_size": 16 if mode == "overfit" else 8,
        },
        "single_gpu_1p5b_pilot_32": {
            "microbatch": 8 if mode == "overfit" else 32,
            "grad_accum": 4 if mode == "overfit" else 2,
            "eval_batch_size": 16 if mode == "overfit" else 8,
        },
        "single_gpu_1p5b_pilot_32_eval1": {
            "microbatch": 8 if mode == "overfit" else 32,
            "grad_accum": 4 if mode == "overfit" else 2,
            "eval_batch_size": 1 if mode == "overfit" else 8,
        },
        "single_gpu_qwen3_4b": {
            "microbatch": 8 if mode == "overfit" else 8,
            "grad_accum": 4 if mode == "overfit" else 8,
            "eval_batch_size": 16 if mode == "overfit" else 1,
        },
    }
    observed = checkpoint.get("training", {})
    if str(checkpoint.get("reconstruction_mode", "base")) != mode:
        raise RuntimeError("checkpoint and data reconstruction modes differ")
    for key, value in expected.items():
        if observed.get(key) != value:
            raise RuntimeError(
                f"checkpoint training field {key!r} is off protocol: "
                f"{observed.get(key)!r} != {value!r}"
            )
    allowed_updates = (
        (300,)
        if mode == "overfit"
        else (pilot_max_updates, 6000, 4000)
    )
    observed_max_updates = observed.get("max_updates")
    if observed_max_updates not in allowed_updates:
        raise RuntimeError(
            "checkpoint training field 'max_updates' is off protocol: "
            f"{observed_max_updates!r} not in {allowed_updates!r}"
        )
    observed_batch_profile = {
        key: observed.get(key)
        for key in ("microbatch", "grad_accum", "eval_batch_size")
    }
    if observed_batch_profile not in batch_profiles.values():
        raise RuntimeError(
            "checkpoint training batch profile is off protocol: "
            f"{observed_batch_profile!r}; expected one of {batch_profiles!r}"
        )
    if int(checkpoint.get("optimizer_update", -1)) != int(observed_max_updates):
        raise RuntimeError("formal evaluation requires the final-update checkpoint")
    prerequisite = checkpoint.get("prerequisite")
    if mode == "pilot" and (
        not isinstance(prerequisite, Mapping)
        or prerequisite.get("decision") != "GO"
    ):
        raise RuntimeError("pilot checkpoint is not authorized by an Overfit GO")
    if mode == "overfit" and prerequisite is not None:
        raise RuntimeError("overfit checkpoint unexpectedly has a prerequisite")


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    dataset: OpenHermesDraftKVReconstructionDataset,
    collator: DraftKVReconstructionCollator,
    donor_mapping: Mapping[str, str],
    generation_ids: Sequence[str],
    output_path: Path,
    *,
    max_new_tokens: int,
) -> list[Dict[str, Any]]:
    existing = {_generation_key(row): row for row in read_jsonl(output_path)}
    expected_keys = {
        (condition, str(example_id))
        for example_id in generation_ids
        for condition in ("matched", "deranged")
    }
    if not set(existing).issubset(expected_keys):
        raise RuntimeError("existing generation rows do not match this evaluation")
    for position, example_id in enumerate(generation_ids, start=1):
        target = dataset.by_id[str(example_id)]
        donor = dataset.by_id[str(donor_mapping[str(example_id)])]
        prompt = torch.tensor(
            [target["receiver_prompt_input_ids"]], dtype=torch.long, device=model.device
        )
        for condition, source in (("matched", target), ("deranged", donor)):
            key = (condition, str(example_id))
            if key in existing:
                continue
            packet = _source_packet(model, collator, [source])
            response = greedy_reconstruct(
                model,
                tokenizer,
                prompt,
                packet=packet,
                disable_communication=False,
                max_new_tokens=int(max_new_tokens),
            )
            existing[key] = make_generation_row(
                condition=condition,
                target=target,
                donor=source,
                response=response,
            )
            ordered = [existing[item] for item in sorted(existing)]
            write_jsonl(output_path, ordered)
        print(f"generation {position}/{len(generation_ids)}", flush=True)
    return [existing[key] for key in sorted(expected_keys)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split", choices=("overfit", "gate_val", "reserve_test"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generation-examples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-reserve-test", action="store_true")
    parser.add_argument("--require-gate-go")
    args = parser.parse_args()
    if args.split == "reserve_test":
        if not args.allow_reserve_test:
            raise RuntimeError("reserve_test requires explicit --allow-reserve-test")
        if not args.require_gate_go:
            raise RuntimeError("reserve_test requires --require-gate-go")
    elif args.allow_reserve_test or args.require_gate_go:
        raise RuntimeError(
            "reserve authorization flags are only valid for reserve_test"
        )
    if (
        args.batch_size <= 0
        or args.generation_examples <= 0
        or args.max_new_tokens < 0
        or args.bootstrap_samples <= 0
    ):
        raise ValueError("batch and generation sizes must be positive")
    seed_all(args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "eval_result.json"
    if result_path.exists():
        raise RuntimeError(
            "evaluation result already exists; use a new output directory"
        )
    manifest, records, manifest_path, records_path = load_reconstruction_bundle(
        args.data_dir
    )
    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(args.device),
        weights_only=True,
    )
    if checkpoint.get("protocol") != PROTOCOL:
        raise RuntimeError("unexpected reconstruction checkpoint protocol")
    _validate_training_protocol(
        checkpoint,
        split=args.split,
        reconstruction_mode=str(manifest.get("reconstruction_mode", "base")),
        expected_seed=int(manifest["seed"]),
    )
    if str(checkpoint["data_manifest_sha256"]) != sha256_file(manifest_path):
        raise RuntimeError("checkpoint and data manifest SHA256 differ")
    if str(checkpoint["records_sha256"]) != sha256_file(records_path):
        raise RuntimeError("checkpoint and reconstruction record SHA256 differ")
    for key in ("receiver", "sharer"):
        if str(checkpoint[key]) != str(manifest[key]):
            raise RuntimeError(f"checkpoint/manifest mismatch for {key}")
    checkpoint_mapping = {
        int(key): int(value) for key, value in checkpoint["layer_mapping"].items()
    }
    manifest_mapping = {
        int(key): int(value) for key, value in manifest["layer_mapping"].items()
    }
    if checkpoint_mapping != manifest_mapping:
        raise RuntimeError("checkpoint/manifest layer mapping mismatch")
    prerequisite = None
    if args.split == "reserve_test":
        prerequisite = validate_go_prerequisite(
            args.require_gate_go,
            expected_split="gate_val",
            data_manifest_path=manifest_path,
            records_path=records_path,
            checkpoint_sha256=checkpoint_sha,
        )

    model, receiver_tokenizer, sharer_tokenizer = build_model(
        receiver_path=str(manifest["receiver"]),
        sharer_path=str(manifest["sharer"]),
        layer_mapping=manifest["layer_mapping"],
        device_name=args.device,
    )
    receiver_prompt = verify_receiver_prompt(receiver_tokenizer, manifest)
    load_trainable_state(model, checkpoint)
    model.set_stage("eval")
    collator = DraftKVReconstructionCollator(
        receiver_tokenizer, sharer_tokenizer
    )

    if args.split == "overfit":
        ids = [str(value) for value in manifest["overfit_ids"]]
        selected = [row for row in records if str(row["example_id"]) in set(ids)]
        dataset_split = "train"
        donor_mapping = make_no_fixed_point_id_mapping(
            ids, seed=int(manifest["derangement_seed"])
        )
    else:
        ids = [str(value) for value in manifest["split_ids"][args.split]]
        selected = [row for row in records if str(row["example_id"]) in set(ids)]
        dataset_split = args.split
        donor_mapping = {
            str(key): str(value)
            for key, value in manifest[
                "gate_derangement"
                if args.split == "gate_val"
                else "reserve_derangement"
            ].items()
        }
    if set(donor_mapping) != set(ids) or any(
        target == donor for target, donor in donor_mapping.items()
    ):
        raise RuntimeError("evaluation donor mapping is not a full derangement")
    dataset = _make_dataset(
        selected,
        receiver_tokenizer,
        sharer_tokenizer,
        manifest,
        split=dataset_split,
    )
    if set(dataset.by_id) != set(ids):
        raise RuntimeError("evaluation dataset IDs differ from manifest")

    static_id = str(manifest["static_donor_id"])
    static_records = [row for row in records if str(row["example_id"]) == static_id]
    static_dataset = _make_dataset(
        static_records,
        receiver_tokenizer,
        sharer_tokenizer,
        manifest,
        split="train",
    )
    static_row = static_dataset[0]
    generation_ids = ids[: min(int(args.generation_examples), len(ids))]
    max_new_tokens = int(args.max_new_tokens) or int(
        manifest["lengths"]["max_message_tokens"]
    ) + 32

    eval_manifest = {
        "protocol": PROTOCOL,
        "reconstruction_mode": str(manifest.get("reconstruction_mode", "base")),
        "split": args.split,
        "data_manifest_sha256": sha256_file(manifest_path),
        "records_sha256": sha256_file(records_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "example_ids": ids,
        "donor_mapping": donor_mapping,
        "static_donor_id": static_id,
        "generation_ids": generation_ids,
        "batch_size": int(args.batch_size),
        "max_new_tokens": max_new_tokens,
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "prerequisite": prerequisite,
    }
    eval_manifest_path = output / "eval_manifest.json"
    if eval_manifest_path.exists():
        old = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
        if old != eval_manifest:
            raise RuntimeError(
                "existing eval manifest differs from requested evaluation"
            )
    else:
        orphaned = [
            path.name
            for path in (
                output / "per_example_nll.jsonl",
                output / "per_example_generation.jsonl",
            )
            if path.exists()
        ]
        if orphaned:
            raise RuntimeError(
                "evaluation artifacts exist without eval_manifest.json: "
                + ", ".join(orphaned)
            )
        write_json(eval_manifest_path, eval_manifest)

    parity = _native_split_parity(model, collator([dataset[0]]))
    if parity > 1e-5:
        raise RuntimeError(f"native/split parity failed: max_abs={parity}")
    nll_rows = evaluate_nll(
        model,
        dataset,
        collator,
        donor_mapping,
        static_row,
        batch_size=args.batch_size,
    )
    write_jsonl(output / "per_example_nll.jsonl", nll_rows)
    generation_rows = evaluate_generation(
        model,
        receiver_tokenizer,
        dataset,
        collator,
        donor_mapping,
        generation_ids,
        output / "per_example_generation.jsonl",
        max_new_tokens=max_new_tokens,
    )
    summary = summarize_reconstruction(
        nll_rows,
        generation_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result = {
        "protocol": PROTOCOL,
        "reconstruction_mode": str(manifest.get("reconstruction_mode", "base")),
        "split": args.split,
        "checkpoint_sha256": checkpoint_sha,
        "eval_manifest_sha256": sha256_file(eval_manifest_path),
        "native_vs_split_max_abs": parity,
        "receiver_prompt": receiver_prompt,
        "prerequisite": prerequisite,
        **summary,
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
