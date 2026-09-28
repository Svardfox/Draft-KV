#!/usr/bin/env python3
"""Stage 3: ARC option training with one-sided Deranged protection.

This stage starts from the Qwen3-0.6B -> Qwen2.5-0.5B OpenHermes
Stage-2 checkpoint and keeps the original ARC data and reconstruction replay
protocol. It uses a one-sided MC objective and selects on Matched accuracy:

    CE(Matched, gold)
    + protection_weight * relu(NLL(Deranged) - stopgrad(NLL(Zero)) - tolerance)

The Receiver and Sharer language models remain frozen.  Only the existing
Draft-KV projection matrices and per-head gates are optimized.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_mc_training_data import (  # noqa: E402
    DraftKVMCOptionCollator,
    DraftKVMCOptionDataset,
    PROTOCOL,
    load_mc_bundle,
    option_classification_stats,
    deranged_no_harm_loss,
    option_token_ids,
)
from draft_kv.train.draft_kv_reconstruction_data import (  # noqa: E402
    DraftKVReconstructionCollator,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    build_model,
    count_parameters,
    load_trainable_state,
    seed_all,
    sha256_file,
    trainable_state,
    write_json,
)
from script.draft_kv.draft_kv_reconstruction_common import (  # noqa: E402
    load_reconstruction_bundle,
)
from script.draft_kv.stage3_common import (  # noqa: E402
    _CyclingLoader,
    _canonical_mapping,
    _matched_deranged_packet_batch,
    _mc_dataset,
    _packet,
    _reconstruction_dataset,
    _train_replay_step,
    _validate_initial_checkpoint,
    mc_validation,
    reconstruction_validation,
)


TRAIN_PROTOCOL = "draft_kv_stage3_mc_checkpoint"
OBJECTIVE = "matched_ce_plus_deranged_no_harm"
DEFAULT_INITIAL_CHECKPOINT = (
    "/workspace/draft-kv/"
    "openhermes_stage2_response_kv_qwen3_to_qwen25_05b_eos_pool/train/last.pt"
)
DEFAULT_RECONSTRUCTION_DATA = (
    "/workspace/draft-kv/"
    "openhermes_reconstruction_qwen3_to_qwen25_05b_longrun/data"
)
DEFAULT_MC_DATA = (
    "/workspace/draft-kv/"
    "stage3_mc_training/data"
)
DEFAULT_OUTPUT = (
    "/workspace/draft-kv/"
    "stage3_mc_training/train"
)


def _train_mc_protection_step(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    cycle: _CyclingLoader,
    dataset: DraftKVMCOptionDataset,
    collator: DraftKVMCOptionCollator,
    derangement: Mapping[str, str],
    option_ids: Sequence[int],
    grad_accum: int,
    protection_weight: float,
    protection_tolerance: float,
    trainable: Sequence[torch.nn.Parameter],
) -> Dict[str, float]:
    """Run one MC update for the Stage 3 objective."""

    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    ce_total = 0.0
    protection_total = 0.0
    matched_correct = 0
    deranged_correct = 0
    zero_correct = 0
    active_total = 0.0
    count = 0

    for _ in range(int(grad_accum)):
        batch = cycle.next()
        receiver_ids = batch["receiver_input_ids"]
        receiver_attention = batch["receiver_attention_mask"]

        # The frozen Receiver-only distribution is the fixed teacher.  Running
        # it without a graph makes the stop-gradient contract explicit.
        with torch.no_grad():
            zero_logits, _, _ = model.forward_receiver(
                receiver_ids,
                receiver_attention,
                packet=None,
                disable_communication=True,
                use_cache=False,
            )
            zero = option_classification_stats(
                zero_logits[:, -1],
                option_ids=option_ids,
                option_counts=batch["option_count"],
                gold_indices=batch["answer_index"],
            )

        packet_batch = _matched_deranged_packet_batch(
            batch,
            dataset=dataset,
            collator=collator,
            derangement=derangement,
        )
        packet = _packet(model, packet_batch, detach=False)
        paired_receiver_ids = torch.cat([receiver_ids, receiver_ids], dim=0)
        paired_receiver_attention = torch.cat(
            [receiver_attention, receiver_attention], dim=0
        )
        logits, _, _ = model.forward_receiver(
            paired_receiver_ids,
            paired_receiver_attention,
            packet=packet,
            disable_communication=False,
            use_cache=False,
        )
        batch_size = len(batch["example_id"])
        matched = option_classification_stats(
            logits[:batch_size, -1],
            option_ids=option_ids,
            option_counts=batch["option_count"],
            gold_indices=batch["answer_index"],
        )
        deranged = option_classification_stats(
            logits[batch_size:, -1],
            option_ids=option_ids,
            option_counts=batch["option_count"],
            gold_indices=batch["answer_index"],
        )

        matched_ce = matched["nll"].mean()
        protection_values = deranged_no_harm_loss(
            deranged["nll"], zero["nll"], tolerance=protection_tolerance
        )
        protection = protection_values.mean()
        active_total += float((protection_values.detach() > 0).float().mean())
        loss = matched_ce + float(protection_weight) * protection
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite Stage 3 MC training loss")
        (loss / float(grad_accum)).backward()

        loss_total += float(loss.detach())
        ce_total += float(matched_ce.detach())
        protection_total += float(protection.detach())
        matched_correct += int(matched["correct"].sum())
        deranged_correct += int(deranged["correct"].sum())
        zero_correct += int(zero["correct"].sum())
        count += int(matched["correct"].numel())

    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    ):
        raise RuntimeError("non-finite Stage 3 communication gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    denominator = float(grad_accum)
    return {
        "loss": loss_total / denominator,
        "matched_option_ce": ce_total / denominator,
        "deranged_protection_loss": protection_total / denominator,
        "protection_active_fraction": active_total / denominator,
        "matched_accuracy": matched_correct / max(1, count),
        "deranged_accuracy": deranged_correct / max(1, count),
        "zero_accuracy": zero_correct / max(1, count),
        "gradient_norm_before_clip": float(gradient_norm),
    }


def _selection_score(calibration: Mapping[str, Any]) -> float:
    """Select exclusively on Matched accuracy; lower NLL breaks exact ties."""
    return float(calibration["conditions"]["matched"]["accuracy"])


def _selection_eligibility(
    calibration: Mapping[str, Any],
    reconstruction_preservation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Preserve reconstruction; report gains without filtering by Zero NLL."""

    gaps = calibration["gaps"]
    gold_probability_gain = float(gaps["matched_minus_zero_gold_log_probability"])
    accuracy_gain = float(gaps["matched_minus_zero_accuracy"])
    matched_beats_zero_nll = gold_probability_gain > 0.0
    reconstruction_eligible = bool(reconstruction_preservation["eligible"])
    return {
        "reconstruction_eligible": reconstruction_eligible,
        "matched_beats_zero_nll": matched_beats_zero_nll,
        "matched_minus_zero_gold_log_probability": gold_probability_gain,
        "matched_minus_zero_accuracy": accuracy_gain,
        "eligible": bool(reconstruction_eligible),
    }


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    optimizer_update: int,
    mc_update: int,
    replay_update: int,
    manifest_path: Path,
    records_path: Path,
    drafts_path: Path,
    initial_checkpoint: Path,
    reconstruction_manifest_path: Path,
    reconstruction_records_path: Path,
    calibration: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    reconstruction_baseline: Mapping[str, Any],
    preservation: Mapping[str, Any],
    selection_eligibility: Mapping[str, Any],
    selection_score: float,
) -> Dict[str, Any]:
    return {
        "protocol": TRAIN_PROTOCOL,
        "objective": OBJECTIVE,
        "data_protocol": PROTOCOL,
        "receiver": str(args.receiver_resolved),
        "sharer": str(args.sharer_resolved),
        "layer_mapping": dict(args.layer_mapping_resolved),
        "communication_state": trainable_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_update": int(optimizer_update),
        "mc_update": int(mc_update),
        "reconstruction_replay_update": int(replay_update),
        "mc_data_manifest_sha256": sha256_file(manifest_path),
        "mc_records_sha256": sha256_file(records_path),
        "mc_drafts_sha256": sha256_file(drafts_path),
        "initial_stage2_checkpoint": str(initial_checkpoint.resolve()),
        "initial_stage2_checkpoint_sha256": sha256_file(initial_checkpoint),
        "reconstruction_manifest_sha256": sha256_file(reconstruction_manifest_path),
        "reconstruction_records_sha256": sha256_file(reconstruction_records_path),
        "calibration": {
            key: value for key, value in calibration.items() if key != "rows"
        },
        "reconstruction_validation": dict(reconstruction),
        "reconstruction_baseline": dict(reconstruction_baseline),
        "reconstruction_preservation": dict(preservation),
        "selection_eligibility": dict(selection_eligibility),
        "selection_score": float(selection_score),
        "selection_criterion": "matched_accuracy_then_lower_matched_nll",
        "training": {
            "max_mc_updates": int(args.max_mc_updates),
            "mc_updates_per_replay": int(args.mc_updates_per_replay),
            "microbatch": int(args.microbatch),
            "grad_accum": int(args.grad_accum),
            "projector_lr": float(args.projector_lr),
            "gate_lr": float(args.gate_lr),
            "protection_weight": float(args.protection_weight),
            "protection_tolerance": float(args.protection_tolerance),
            "seed": int(args.seed),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Draft-KV Stage 3 with Matched answer CE and "
            "one-sided Deranged answer-NLL protection."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_MC_DATA)
    parser.add_argument(
        "--reconstruction-data-dir", default=DEFAULT_RECONSTRUCTION_DATA
    )
    parser.add_argument(
        "--initial-stage2-checkpoint", default=DEFAULT_INITIAL_CHECKPOINT
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-mc-updates", type=int, default=4000)
    parser.add_argument("--mc-updates-per-replay", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--projector-lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=5e-4)
    parser.add_argument("--protection-weight", type=float, default=0.1)
    parser.add_argument("--protection-tolerance", type=float, default=0.1)
    parser.add_argument("--eval-every-mc-updates", type=int, default=250)
    parser.add_argument("--log-every-mc-updates", type=int, default=10)
    parser.add_argument("--max-reconstruction-nll-increase", type=float, default=0.10)
    parser.add_argument("--min-reconstruction-gap-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=31847)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    positive = (
        args.max_mc_updates,
        args.mc_updates_per_replay,
        args.microbatch,
        args.grad_accum,
        args.eval_batch_size,
        args.eval_every_mc_updates,
        args.log_every_mc_updates,
    )
    if min(positive) <= 0 or args.projector_lr <= 0 or args.gate_lr <= 0:
        raise ValueError("training sizes and learning rates must be positive")
    if not math.isfinite(args.protection_weight) or args.protection_weight < 0:
        raise ValueError("protection weight must be finite and nonnegative")
    if not math.isfinite(args.protection_tolerance) or args.protection_tolerance < 0:
        raise ValueError("protection tolerance must be finite and nonnegative")
    if args.max_reconstruction_nll_increase < 0:
        raise ValueError("reconstruction NLL allowance cannot be negative")
    if not 0 <= args.min_reconstruction_gap_fraction <= 1:
        raise ValueError("minimum reconstruction gap fraction must be in [0,1]")
    seed_all(args.seed)

    output = Path(args.output_dir).resolve()
    artifacts = (
        output / "last.pt",
        output / "best.pt",
        output / "train_history.json",
        output / "train_result.json",
    )
    if any(path.exists() for path in artifacts):
        raise RuntimeError("training output exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)

    (
        manifest,
        records,
        drafts,
        manifest_path,
        records_path,
        drafts_path,
    ) = load_mc_bundle(args.data_dir)
    (
        reconstruction_manifest,
        reconstruction_records,
        reconstruction_manifest_path,
        reconstruction_records_path,
    ) = load_reconstruction_bundle(args.reconstruction_data_dir)
    initial_path = Path(args.initial_stage2_checkpoint).resolve()
    checkpoint = torch.load(initial_path, map_location="cpu", weights_only=True)
    _validate_initial_checkpoint(checkpoint, path=initial_path, manifest=manifest)
    expected_reconstruction_sha = checkpoint.get("stage1_data_manifest_sha256")
    if expected_reconstruction_sha is not None and str(
        expected_reconstruction_sha
    ) != sha256_file(reconstruction_manifest_path):
        raise RuntimeError(
            "OpenHermes replay manifest differs from the Stage-2 checkpoint lineage"
        )

    model, receiver_tokenizer, sharer_tokenizer = build_model(
        receiver_path=str(manifest["receiver"]),
        sharer_path=str(manifest["sharer"]),
        layer_mapping=manifest["layer_mapping"],
        device_name=args.device,
    )
    load_trainable_state(model, checkpoint)
    args.receiver_resolved = str(manifest["receiver"])
    args.sharer_resolved = str(manifest["sharer"])
    args.layer_mapping_resolved = _canonical_mapping(manifest["layer_mapping"])

    mc_collator = DraftKVMCOptionCollator(receiver_tokenizer, sharer_tokenizer)
    train_dataset = _mc_dataset(
        manifest,
        records,
        drafts,
        receiver_tokenizer,
        sharer_tokenizer,
        split="train",
    )
    calibration_dataset = _mc_dataset(
        manifest,
        records,
        drafts,
        receiver_tokenizer,
        sharer_tokenizer,
        split="calibration",
    )
    replay_train = _reconstruction_dataset(
        reconstruction_manifest,
        reconstruction_records,
        receiver_tokenizer,
        sharer_tokenizer,
        split="train",
    )
    replay_gate = _reconstruction_dataset(
        reconstruction_manifest,
        reconstruction_records,
        receiver_tokenizer,
        sharer_tokenizer,
        split="gate_val",
    )
    replay_collator = DraftKVReconstructionCollator(
        receiver_tokenizer, sharer_tokenizer
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.microbatch),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed)),
        num_workers=0,
        collate_fn=mc_collator,
    )
    replay_loader = DataLoader(
        replay_train,
        batch_size=int(args.microbatch),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed) + 1),
        num_workers=0,
        collate_fn=replay_collator,
    )
    replay_gate_loader = DataLoader(
        replay_gate,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=replay_collator,
    )
    options = option_token_ids(receiver_tokenizer)
    initial_calibration = mc_validation(
        model,
        calibration_dataset,
        mc_collator,
        derangement=manifest["calibration_derangement"],
        option_ids=options,
        batch_size=args.eval_batch_size,
    )
    reconstruction_baseline = reconstruction_validation(model, replay_gate_loader)
    history: list[Dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "mc_update": 0,
            "calibration": {
                key: value
                for key, value in initial_calibration.items()
                if key != "rows"
            },
            "reconstruction_validation": reconstruction_baseline,
        }
    ]
    print("validation", json.dumps(history[-1]), flush=True)

    model.set_stage("communication")
    if any(parameter.requires_grad for parameter in model.receiver.parameters()):
        raise RuntimeError("Receiver LLM unexpectedly has trainable parameters")
    if any(parameter.requires_grad for parameter in model.sharer.parameters()):
        raise RuntimeError("Sharer LLM unexpectedly has trainable parameters")
    names = model.trainable_parameter_names()
    if any(
        not (name.startswith("projection.") or "gate_logits" in name) for name in names
    ):
        raise RuntimeError(f"unexpected trainable parameters: {names}")
    projector_parameters = list(model.projection.parameters())
    gate_parameters = list(model.consumer.gate_parameters)
    trainable = list(model.communication_parameters)
    parameter_counts = {
        "projector": count_parameters(projector_parameters),
        "gate": count_parameters(gate_parameters),
        "total": count_parameters(trainable),
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": projector_parameters, "lr": float(args.projector_lr)},
            {"params": gate_parameters, "lr": float(args.gate_lr)},
        ],
        weight_decay=0.0,
    )
    mc_cycle = _CyclingLoader(train_loader)
    replay_cycle = _CyclingLoader(replay_loader)
    optimizer_update = 0
    replay_update = 0
    best_score = float("-inf")
    best_matched_nll = float("inf")
    best_mc_update: int | None = None
    start_time = time.time()

    for mc_update in range(1, int(args.max_mc_updates) + 1):
        model.set_stage("communication")
        optimizer_update += 1
        metrics = _train_mc_protection_step(
            model=model,
            optimizer=optimizer,
            cycle=mc_cycle,
            dataset=train_dataset,
            collator=mc_collator,
            derangement=manifest["train_derangement"],
            option_ids=options,
            grad_accum=args.grad_accum,
            protection_weight=args.protection_weight,
            protection_tolerance=args.protection_tolerance,
            trainable=trainable,
        )
        if mc_update == 1 or mc_update % int(args.log_every_mc_updates) == 0:
            row = {
                "optimizer_update": optimizer_update,
                "mc_update": mc_update,
                "objective": OBJECTIVE,
                **metrics,
            }
            history.append(row)
            print("train", json.dumps(row), flush=True)

        if mc_update % int(args.mc_updates_per_replay) == 0:
            model.set_stage("communication")
            optimizer_update += 1
            replay_update += 1
            replay_metrics = _train_replay_step(
                model=model,
                optimizer=optimizer,
                cycle=replay_cycle,
                grad_accum=args.grad_accum,
                trainable=trainable,
            )
            row = {
                "optimizer_update": optimizer_update,
                "mc_update": mc_update,
                "reconstruction_replay_update": replay_update,
                "objective": "openhermes_reconstruction_replay",
                **replay_metrics,
            }
            history.append(row)
            print("train", json.dumps(row), flush=True)

        if mc_update % int(args.eval_every_mc_updates) != 0 and mc_update != int(
            args.max_mc_updates
        ):
            continue

        calibration = mc_validation(
            model,
            calibration_dataset,
            mc_collator,
            derangement=manifest["calibration_derangement"],
            option_ids=options,
            batch_size=args.eval_batch_size,
        )
        reconstruction = reconstruction_validation(model, replay_gate_loader)
        max_matched = float(reconstruction_baseline["matched_mean_nll"]) * (
            1.0 + float(args.max_reconstruction_nll_increase)
        )
        min_gap = float(reconstruction_baseline["zero_minus_matched_nll"]) * float(
            args.min_reconstruction_gap_fraction
        )
        preservation = {
            "matched_nll_within_limit": (
                float(reconstruction["matched_mean_nll"]) <= max_matched
            ),
            "causal_gap_within_limit": (
                float(reconstruction["zero_minus_matched_nll"]) >= min_gap
            ),
            "max_matched_nll": max_matched,
            "min_zero_minus_matched_nll": min_gap,
        }
        preservation["eligible"] = bool(
            preservation["matched_nll_within_limit"]
            and preservation["causal_gap_within_limit"]
        )
        selection_eligibility = _selection_eligibility(calibration, preservation)
        score = _selection_score(calibration)
        matched_nll = float(calibration["conditions"]["matched"]["mean_nll"])
        row = {
            "optimizer_update": optimizer_update,
            "mc_update": mc_update,
            "reconstruction_replay_update": replay_update,
            "calibration": {
                key: value for key, value in calibration.items() if key != "rows"
            },
            "reconstruction_validation": reconstruction,
            "reconstruction_preservation": preservation,
            "selection_eligibility": selection_eligibility,
            "selection_score": score,
        }
        history.append(row)
        print("validation", json.dumps(row), flush=True)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            args=args,
            optimizer_update=optimizer_update,
            mc_update=mc_update,
            replay_update=replay_update,
            manifest_path=manifest_path,
            records_path=records_path,
            drafts_path=drafts_path,
            initial_checkpoint=initial_path,
            reconstruction_manifest_path=reconstruction_manifest_path,
            reconstruction_records_path=reconstruction_records_path,
            calibration=calibration,
            reconstruction=reconstruction,
            reconstruction_baseline=reconstruction_baseline,
            preservation=preservation,
            selection_eligibility=selection_eligibility,
            selection_score=score,
        )
        torch.save(payload, output / "last.pt")
        if selection_eligibility["eligible"] and (score, -matched_nll) > (
            best_score, -best_matched_nll
        ):
            best_score = score
            best_matched_nll = matched_nll
            best_mc_update = mc_update
            torch.save(payload, output / "best.pt")
        write_json(output / "train_history.json", history)

    result = {
        "protocol": TRAIN_PROTOCOL,
        "objective": OBJECTIVE,
        "data_protocol": PROTOCOL,
        "initial_stage2_checkpoint": str(initial_path),
        "initial_stage2_checkpoint_sha256": sha256_file(initial_path),
        "best_checkpoint": (
            str((output / "best.pt").resolve()) if best_mc_update is not None else None
        ),
        "best_checkpoint_sha256": (
            sha256_file(output / "best.pt") if best_mc_update is not None else None
        ),
        "last_checkpoint": str((output / "last.pt").resolve()),
        "last_checkpoint_sha256": sha256_file(output / "last.pt"),
        "best_mc_update": best_mc_update,
        "best_selection_score": (best_score if best_mc_update is not None else None),
        "best_matched_nll": (best_matched_nll if best_mc_update is not None else None),
        "selection_criterion": "matched_accuracy_then_lower_matched_nll",
        "mc_updates": int(args.max_mc_updates),
        "reconstruction_replay_updates": replay_update,
        "optimizer_updates": optimizer_update,
        "replay_schedule": (
            f"{int(args.mc_updates_per_replay)} MC updates followed by "
            "1 OpenHermes reconstruction replay update"
        ),
        "protection_weight": float(args.protection_weight),
        "protection_tolerance": float(args.protection_tolerance),
        "mc_train_examples": len(train_dataset),
        "calibration_examples": len(calibration_dataset),
        "reconstruction_train_examples": len(replay_train),
        "reconstruction_gate_examples": len(replay_gate),
        "parameter_counts": parameter_counts,
        "trainable_parameter_names": list(names),
        "runtime_seconds": time.time() - start_time,
    }
    write_json(output / "train_history.json", history)
    write_json(output / "train_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if best_mc_update is None:
        raise RuntimeError("No reconstruction-eligible checkpoint; benchmark evaluation must not run")


if __name__ == "__main__":
    main()
