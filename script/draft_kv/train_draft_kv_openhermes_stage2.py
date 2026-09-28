"""Stage-2 answer training with Stage-1 reconstruction replay.

Both language models remain frozen.  The only optimized parameters are the
existing K/V projection matrices and one scalar gate per Receiver KV head at
each mapped layer, loaded from the registered Stage-1 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_openhermes_stage2_data import (  # noqa: E402
    DraftKVStage2Collator,
    PROTOCOL,
)
from draft_kv.train.draft_kv_reconstruction_data import (  # noqa: E402
    DraftKVReconstructionCollator,
    OpenHermesDraftKVReconstructionDataset,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    answer_nll,
    build_model,
    count_parameters,
    load_trainable_state,
    seed_all,
    sha256_file,
    trainable_state,
    write_json,
)
from script.draft_kv.draft_kv_openhermes_stage2_common import (  # noqa: E402
    condition_nll_rows,
    load_stage2_bundle,
    make_stage2_dataset,
    parity_checks,
    summarize_condition_rows,
)
from script.draft_kv.draft_kv_reconstruction_common import (  # noqa: E402
    load_reconstruction_bundle,
    token_reconstruction_stats,
)


class _CyclingLoader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iterator: Iterator[Mapping[str, Any]] = iter(loader)

    def next(self) -> Mapping[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def _canonical_mapping(value: Mapping[Any, Any]) -> Dict[int, int]:
    return {int(target): int(source) for target, source in value.items()}


def _validate_stage1_checkpoint(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    manifest: Mapping[str, Any],
    expected_sha256: str,
) -> None:
    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != str(expected_sha256):
        raise RuntimeError(
            f"Stage-1 checkpoint SHA mismatch: expected {expected_sha256}, got {observed_sha}"
        )
    if checkpoint.get("protocol") != "draft_kv_openhermes_reconstruction":
        raise RuntimeError("Stage-1 initialization is not a reconstruction checkpoint")
    for name in ("receiver", "sharer"):
        if Path(str(checkpoint[name])).resolve() != Path(str(manifest[name])).resolve():
            raise RuntimeError(f"Stage-1 {name} differs from Stage-2 manifest")
    if _canonical_mapping(checkpoint["layer_mapping"]) != _canonical_mapping(
        manifest["layer_mapping"]
    ):
        raise RuntimeError("Stage-1 layer mapping differs from Stage 2")


def _reconstruction_dataset(
    manifest: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
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
def reconstruction_validation(model: Any, loader: DataLoader) -> Dict[str, float]:
    model.set_stage("eval")
    values = {"zero": [], "matched": []}
    correct = {"zero": 0, "matched": 0}
    counts = {"zero": 0, "matched": 0}
    for batch in loader:
        packet = model.make_packet(
            batch["sharer_input_ids"],
            batch["sharer_attention_mask"],
            batch["sharer_draft_mask"],
            detach=True,
        )
        for condition in ("zero", "matched"):
            logits, _, _ = model.forward_receiver(
                batch["receiver_input_ids"],
                batch["receiver_attention_mask"],
                packet=packet if condition == "matched" else None,
                disable_communication=condition == "zero",
                use_cache=False,
            )
            stats = token_reconstruction_stats(logits, batch["labels"])
            values[condition].extend(float(value) for value in stats["mean"])
            correct[condition] += int(stats["correct"].sum())
            counts[condition] += int(stats["count"].sum())
    zero = float(np.mean(values["zero"]))
    matched = float(np.mean(values["matched"]))
    return {
        "zero_mean_nll": zero,
        "matched_mean_nll": matched,
        "zero_minus_matched_nll": zero - matched,
        "zero_token_accuracy": correct["zero"] / max(1, counts["zero"]),
        "matched_token_accuracy": correct["matched"] / max(1, counts["matched"]),
        "count": len(values["zero"]),
    }


def _answer_validation(
    model: Any,
    dataset: Any,
    packet_rows: Mapping[str, Mapping[str, Any]],
    collator: DraftKVStage2Collator,
    manifest: Mapping[str, Any],
    *,
    batch_size: int,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    rows = condition_nll_rows(
        model,
        dataset,
        packet_rows,
        collator,
        derangement=manifest["gate_derangement"],
        static_donor_id=str(manifest["static_donor_id"]),
        batch_size=batch_size,
    )
    return summarize_condition_rows(rows), rows


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    protocol: str,
    update: int,
    stage2_manifest_path: Path,
    stage2_records_path: Path,
    stage2_drafts_path: Path,
    stage1_checkpoint_path: Path,
    stage1_manifest_path: Path,
    answer_validation: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    reconstruction_baseline: Mapping[str, Any],
    preservation: Mapping[str, Any],
) -> Dict[str, Any]:
    training = {
        "max_updates": int(args.max_updates),
        "microbatch": int(args.microbatch),
        "grad_accum": int(args.grad_accum),
        "reconstruction_replay_every": int(args.reconstruction_replay_every),
        "projector_lr": float(args.projector_lr),
        "gate_lr": float(args.gate_lr),
        "seed": int(args.seed),
    }
    if int(args.snapshot_every) > 0:
        training["snapshot_every"] = int(args.snapshot_every)
    return {
        "protocol": str(protocol),
        "receiver": str(args.receiver_resolved),
        "sharer": str(args.sharer_resolved),
        "layer_mapping": dict(args.layer_mapping_resolved),
        "communication_state": trainable_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_update": int(update),
        "stage2_data_manifest_sha256": sha256_file(stage2_manifest_path),
        "stage2_records_sha256": sha256_file(stage2_records_path),
        "stage2_drafts_sha256": sha256_file(stage2_drafts_path),
        "stage1_checkpoint": str(stage1_checkpoint_path.resolve()),
        "stage1_checkpoint_sha256": sha256_file(stage1_checkpoint_path),
        "stage1_data_manifest_sha256": sha256_file(stage1_manifest_path),
        "answer_validation": dict(answer_validation),
        "reconstruction_validation": dict(reconstruction),
        "reconstruction_baseline": dict(reconstruction_baseline),
        "reconstruction_preservation": dict(preservation),
        "training": training,
    }


def _snapshot_path(output: Path, update: int) -> Path:
    return output / "snapshots" / f"update_{int(update):08d}.pt"


def _should_save_snapshot(update: int, snapshot_every: int) -> bool:
    return int(snapshot_every) > 0 and int(update) % int(snapshot_every) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--stage1-data-dir", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-updates", type=int, default=4000)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--reconstruction-replay-every", type=int, default=5)
    parser.add_argument("--projector-lr", type=float, default=2e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="save snapshots/update_XXXXXXXX.pt every N updates; 0 disables",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-reconstruction-nll-increase", type=float, default=0.10)
    parser.add_argument("--min-reconstruction-gap-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    numeric = (
        args.max_updates,
        args.microbatch,
        args.grad_accum,
        args.eval_batch_size,
        args.reconstruction_replay_every,
        args.eval_every,
        args.log_every,
    )
    if min(numeric) <= 0 or args.projector_lr <= 0 or args.gate_lr <= 0:
        raise ValueError("training sizes and learning rates must be positive")
    if args.snapshot_every < 0:
        raise ValueError("--snapshot-every cannot be negative")
    if args.max_reconstruction_nll_increase < 0:
        raise ValueError("max reconstruction NLL increase cannot be negative")
    if not 0 <= args.min_reconstruction_gap_fraction <= 1:
        raise ValueError("min reconstruction gap fraction must be in [0,1]")
    seed_all(args.seed)

    output = Path(args.output_dir)
    artifacts = (
        output / "last.pt",
        output / "best.pt",
        output / "train_history.json",
        output / "train_result.json",
    )
    if any(path.exists() for path in artifacts):
        raise RuntimeError("Stage 2 training output exists; use a new directory")
    if args.snapshot_every and (output / "snapshots").exists():
        raise RuntimeError("Stage 2 snapshot output exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.snapshot_every:
        (output / "snapshots").mkdir(parents=True, exist_ok=False)
    (
        manifest,
        records,
        drafts,
        manifest_path,
        records_path,
        drafts_path,
    ) = load_stage2_bundle(args.data_dir)
    stage1_manifest, stage1_records, stage1_manifest_path, stage1_records_path = (
        load_reconstruction_bundle(args.stage1_data_dir)
    )
    checkpoint_path = Path(args.stage1_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_stage1_checkpoint(
        checkpoint,
        checkpoint_path,
        manifest=manifest,
        expected_sha256=args.expected_stage1_checkpoint_sha256,
    )
    if checkpoint.get("data_manifest_sha256") != sha256_file(stage1_manifest_path):
        raise RuntimeError("Stage-1 checkpoint and replay manifest differ")
    if checkpoint.get("records_sha256") != sha256_file(stage1_records_path):
        raise RuntimeError("Stage-1 checkpoint and replay records differ")
    if manifest.get("stage1_manifest_sha256") != sha256_file(stage1_manifest_path):
        raise RuntimeError("Stage-2 exclusion manifest and replay manifest differ")
    if int(manifest["command_config"]["seed"]) != int(args.seed):
        raise RuntimeError("Stage-2 training seed differs from data manifest")

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

    stage2_collator = DraftKVStage2Collator(receiver_tokenizer, sharer_tokenizer)
    answer_train = make_stage2_dataset(
        manifest, records, drafts, receiver_tokenizer, sharer_tokenizer, split="train"
    )
    answer_gate = make_stage2_dataset(
        manifest, records, drafts, receiver_tokenizer, sharer_tokenizer, split="gate_val"
    )
    packet_rows = dict(answer_train.by_id) | dict(answer_gate.by_id)
    replay_train = _reconstruction_dataset(
        stage1_manifest,
        stage1_records,
        receiver_tokenizer,
        sharer_tokenizer,
        split="train",
    )
    replay_gate = _reconstruction_dataset(
        stage1_manifest,
        stage1_records,
        receiver_tokenizer,
        sharer_tokenizer,
        split="gate_val",
    )
    replay_collator = DraftKVReconstructionCollator(
        receiver_tokenizer, sharer_tokenizer
    )
    generator_answer = torch.Generator().manual_seed(int(args.seed))
    generator_replay = torch.Generator().manual_seed(int(args.seed) + 1)
    answer_loader = DataLoader(
        answer_train,
        batch_size=int(args.microbatch),
        shuffle=True,
        generator=generator_answer,
        num_workers=0,
        collate_fn=stage2_collator,
    )
    replay_loader = DataLoader(
        replay_train,
        batch_size=int(args.microbatch),
        shuffle=True,
        generator=generator_replay,
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
    first_batch = next(iter(answer_loader))
    parity = parity_checks(model, first_batch)
    if parity["native_vs_split_max_abs"] > 1e-5 or parity["base_vs_zero_gate_max_abs"] != 0.0:
        raise RuntimeError(f"Stage 2 parity failed: {parity}")

    initial_answer, _ = _answer_validation(
        model,
        answer_gate,
        packet_rows,
        stage2_collator,
        manifest,
        batch_size=args.eval_batch_size,
    )
    reconstruction_baseline = reconstruction_validation(model, replay_gate_loader)
    history: list[Dict[str, Any]] = [
        {
            "optimizer_update": 0,
            "answer_validation": initial_answer,
            "reconstruction_validation": reconstruction_baseline,
        }
    ]
    print("validation", json.dumps(history[-1]), flush=True)

    model.set_stage("communication")
    projector_parameters = list(model.projection.parameters())
    gate_parameters = list(model.consumer.gate_parameters)
    trainable = list(model.communication_parameters)
    names = model.trainable_parameter_names()
    if any(not (name.startswith("projection.") or "gate_logits" in name) for name in names):
        raise RuntimeError(f"unexpected Stage 2 trainable parameters: {names}")
    parameter_counts = {
        "projector": count_parameters(projector_parameters),
        "gate": count_parameters(gate_parameters),
        "total": count_parameters(trainable),
    }
    # There is one signed gate per Receiver KV head at each mapped layer.
    # The expected count follows the receiver configuration so different pairs
    # share the same Stage-2 trainer.
    expected_gate_count = len(manifest["layer_mapping"]) * int(
        model.receiver.config.num_key_value_heads
    )
    if parameter_counts["gate"] != expected_gate_count:
        raise RuntimeError(f"unexpected Stage 2 parameter counts: {parameter_counts}")
    if parameter_counts["total"] != (
        parameter_counts["projector"] + parameter_counts["gate"]
    ):
        raise RuntimeError(
            f"Stage 2 trainable parameter total is inconsistent: {parameter_counts}"
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": projector_parameters, "lr": float(args.projector_lr)},
            {"params": gate_parameters, "lr": float(args.gate_lr)},
        ],
        weight_decay=0.0,
    )
    answer_cycle = _CyclingLoader(answer_loader)
    replay_cycle = _CyclingLoader(replay_loader)
    best_score = float("-inf")
    best_update: int | None = None
    objective_updates = {"answer": 0, "reconstruction_replay": 0}
    start_time = time.time()

    for update in range(1, int(args.max_updates) + 1):
        objective = (
            "reconstruction_replay"
            if update % int(args.reconstruction_replay_every) == 0
            else "answer"
        )
        objective_updates[objective] += 1
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        token_correct = 0
        token_count = 0
        diagnostics: Mapping[str, Any] = {}
        for microstep in range(int(args.grad_accum)):
            batch = replay_cycle.next() if objective == "reconstruction_replay" else answer_cycle.next()
            packet = model.make_packet(
                batch["sharer_input_ids"],
                batch["sharer_attention_mask"],
                batch["sharer_draft_mask"],
                detach=False,
            )
            logits, _, diagnostics = model.forward_receiver(
                batch["receiver_input_ids"],
                batch["receiver_attention_mask"],
                packet=packet,
                disable_communication=False,
                return_diagnostics=microstep == 0 and update == 1,
                use_cache=False,
            )
            stats = (
                token_reconstruction_stats(logits, batch["labels"])
                if objective == "reconstruction_replay"
                else answer_nll(logits, batch["labels"])
            )
            loss = stats["sum"].sum() / stats["count"].sum().clamp_min(1)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite {objective} loss at update {update}")
            (loss / float(args.grad_accum)).backward()
            loss_sum += float(loss.detach())
            token_count += int(stats["count"].sum())
            if "correct" in stats:
                token_correct += int(stats["correct"].sum())
            else:
                shifted_labels = batch["labels"][:, 1:].to(logits.device)
                valid = shifted_labels.ne(-100)
                predictions = logits[:, :-1].argmax(dim=-1)
                token_correct += int((predictions.eq(shifted_labels) & valid).sum())
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
            raise RuntimeError(f"non-finite communication gradient at update {update}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if update == 1 or update % int(args.log_every) == 0:
            row = {
                "optimizer_update": update,
                "objective": objective,
                "loss": loss_sum / float(args.grad_accum),
                "token_accuracy": token_correct / max(1, token_count),
                "gradient_norm_before_clip": float(gradient_norm),
                "diagnostic_layers": sorted(diagnostics),
            }
            history.append(row)
            print("train", json.dumps(row), flush=True)
        snapshot_due = _should_save_snapshot(update, args.snapshot_every)
        if (
            update % int(args.eval_every) != 0
            and update != int(args.max_updates)
            and not snapshot_due
        ):
            continue

        answer_validation, _ = _answer_validation(
            model,
            answer_gate,
            packet_rows,
            stage2_collator,
            manifest,
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
            "matched_nll_within_limit": float(reconstruction["matched_mean_nll"]) <= max_matched,
            "causal_gap_within_limit": float(reconstruction["zero_minus_matched_nll"]) >= min_gap,
            "max_matched_nll": max_matched,
            "min_zero_minus_matched_nll": min_gap,
        }
        preservation["eligible"] = bool(
            preservation["matched_nll_within_limit"]
            and preservation["causal_gap_within_limit"]
        )
        zero_gap = float(answer_validation["zero"]["mean_nll"]) - float(
            answer_validation["matched"]["mean_nll"]
        )
        deranged_gap = float(answer_validation["deranged"]["mean_nll"]) - float(
            answer_validation["matched"]["mean_nll"]
        )
        score = min(zero_gap, deranged_gap)
        row = {
            "optimizer_update": update,
            "answer_validation": answer_validation,
            "answer_zero_minus_matched_nll": zero_gap,
            "answer_deranged_minus_matched_nll": deranged_gap,
            "reconstruction_validation": reconstruction,
            "reconstruction_preservation": preservation,
            "checkpoint_score": score,
        }
        history.append(row)
        print("validation", json.dumps(row), flush=True)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            args=args,
            protocol=str(manifest["protocol"]),
            update=update,
            stage2_manifest_path=manifest_path,
            stage2_records_path=records_path,
            stage2_drafts_path=drafts_path,
            stage1_checkpoint_path=checkpoint_path,
            stage1_manifest_path=stage1_manifest_path,
            answer_validation=answer_validation,
            reconstruction=reconstruction,
            reconstruction_baseline=reconstruction_baseline,
            preservation=preservation,
        )
        torch.save(payload, output / "last.pt")
        if snapshot_due:
            snapshot_path = _snapshot_path(output, update)
            torch.save(payload, snapshot_path)
            print(f"snapshot {snapshot_path}", flush=True)
        if preservation["eligible"] and score > 0 and score > best_score:
            best_score = score
            best_update = update
            torch.save(payload, output / "best.pt")
        model.set_stage("communication")

    result = {
        "protocol": str(manifest["protocol"]),
        "stage1_checkpoint": str(checkpoint_path.resolve()),
        "stage1_checkpoint_sha256": sha256_file(checkpoint_path),
        "best_checkpoint": str((output / "best.pt").resolve()) if best_update is not None else None,
        "best_checkpoint_sha256": sha256_file(output / "best.pt") if best_update is not None else None,
        "last_checkpoint": str((output / "last.pt").resolve()),
        "last_checkpoint_sha256": sha256_file(output / "last.pt"),
        "best_update": best_update,
        "best_score_min_causal_nll_gap": best_score if best_update is not None else None,
        "parity": parity,
        "optimizer_updates": int(args.max_updates),
        "objective_updates": objective_updates,
        "answer_train_examples": len(answer_train),
        "answer_gate_examples": len(answer_gate),
        "reconstruction_train_examples": len(replay_train),
        "reconstruction_gate_examples": len(replay_gate),
        "projector_parameter_count": parameter_counts["projector"],
        "gate_parameter_count": parameter_counts["gate"],
        "total_trainable_parameter_count": parameter_counts["total"],
        "trainable_parameter_names": list(names),
        "runtime_seconds": time.time() - start_time,
        "snapshot_every": int(args.snapshot_every),
        "snapshots": [
            {
                "optimizer_update": update,
                "path": str(_snapshot_path(output, update).resolve()),
                "sha256": sha256_file(_snapshot_path(output, update)),
            }
            for update in range(
                int(args.snapshot_every),
                int(args.max_updates) + 1,
                int(args.snapshot_every),
            )
        ]
        if args.snapshot_every
        else [],
    }
    write_json(output / "train_history.json", history)
    write_json(output / "train_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
