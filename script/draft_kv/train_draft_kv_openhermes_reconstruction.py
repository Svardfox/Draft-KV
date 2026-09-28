"""Train Draft-KV projectors to reconstruct OpenHermes assistant messages."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_reconstruction_data import (  # noqa: E402
    DraftKVReconstructionCollator,
    OpenHermesDraftKVReconstructionDataset,
    PROTOCOL,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    build_model,
    count_parameters,
    seed_all,
    sha256_file,
    trainable_state,
    write_json,
)
from script.draft_kv.draft_kv_reconstruction_common import (  # noqa: E402
    load_reconstruction_bundle,
    token_reconstruction_stats,
    validate_go_prerequisite,
    verify_receiver_prompt,
)


@torch.no_grad()
def parity_checks(model: Any, batch: Mapping[str, Any]) -> Dict[str, float]:
    model.set_stage("eval")
    ids = batch["receiver_input_ids"].to(model.device)
    mask = batch["receiver_attention_mask"].to(model.device)
    native = model.receiver(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
    ).logits
    base, _, _ = model.forward_receiver(
        ids,
        mask,
        packet=None,
        disable_communication=True,
        use_cache=False,
    )
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
    return {
        "native_vs_split_max_abs": float((native - base).abs().max()),
        "base_vs_zero_gate_max_abs": float((base - zero_gate).abs().max()),
    }


@torch.no_grad()
def validation_nll(model: Any, loader: DataLoader) -> Dict[str, float]:
    model.set_stage("eval")
    rows: Dict[str, list[float]] = {"zero": [], "matched": []}
    token_correct = {"zero": 0, "matched": 0}
    token_count = {"zero": 0, "matched": 0}
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
            rows[condition].extend(float(value) for value in stats["mean"])
            token_correct[condition] += int(stats["correct"].sum())
            token_count[condition] += int(stats["count"].sum())
    result = {
        "zero_example_mean_nll": float(np.mean(rows["zero"])),
        "matched_example_mean_nll": float(np.mean(rows["matched"])),
        "zero_minus_matched_nll": float(
            np.mean(rows["zero"]) - np.mean(rows["matched"])
        ),
        "zero_token_accuracy": token_correct["zero"] / max(1, token_count["zero"]),
        "matched_token_accuracy": token_correct["matched"]
        / max(1, token_count["matched"]),
        "count": len(rows["zero"]),
    }
    if any(
        not np.isfinite(value)
        for key, value in result.items()
        if key != "count"
    ):
        raise RuntimeError("validation produced non-finite reconstruction metrics")
    return result


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


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    records_path: Path,
    args: argparse.Namespace,
    update: int,
    validation: Mapping[str, Any],
    prerequisite: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "reconstruction_mode": str(manifest.get("reconstruction_mode", "base")),
        "receiver": manifest["receiver"],
        "sharer": manifest["sharer"],
        "layer_mapping": manifest["layer_mapping"],
        "data_manifest_sha256": sha256_file(manifest_path),
        "records_sha256": sha256_file(records_path),
        "communication_state": trainable_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_update": int(update),
        "validation": dict(validation),
        "prerequisite": dict(prerequisite) if prerequisite is not None else None,
        "training": {
            "mode": str(args.mode),
            "max_updates": int(args.max_updates),
            "projector_lr": float(args.projector_lr),
            "gate_lr": float(args.gate_lr),
            "microbatch": int(args.microbatch),
            "grad_accum": int(args.grad_accum),
            "eval_batch_size": int(args.eval_batch_size),
            "eval_every": int(args.eval_every),
            "log_every": int(args.log_every),
            "seed": int(args.seed),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("overfit", "pilot"), required=True)
    parser.add_argument("--max-updates", type=int, default=200)
    parser.add_argument("--projector-lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-overfit-go")
    args = parser.parse_args()
    if (
        args.max_updates <= 0
        or args.microbatch <= 0
        or args.grad_accum <= 0
        or args.eval_batch_size <= 0
        or args.eval_every <= 0
        or args.log_every <= 0
        or args.projector_lr <= 0
        or args.gate_lr <= 0
    ):
        raise ValueError("updates and batch parameters must be positive")
    seed_all(args.seed)

    output = Path(args.output_dir)
    training_artifacts = (
        output / "last.pt",
        output / "best.pt",
        output / "train_history.json",
        output / "train_result.json",
    )
    if any(path.exists() for path in training_artifacts):
        raise RuntimeError("training output already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest, records, manifest_path, records_path = load_reconstruction_bundle(
        args.data_dir
    )
    if int(manifest["seed"]) != int(args.seed):
        raise RuntimeError("training seed differs from prepared manifest")
    if args.mode == "pilot":
        if not args.require_overfit_go:
            raise RuntimeError("pilot mode requires --require-overfit-go")
        prerequisite = validate_go_prerequisite(
            args.require_overfit_go,
            expected_split="overfit",
            data_manifest_path=manifest_path,
            records_path=records_path,
        )
    else:
        if args.require_overfit_go:
            raise RuntimeError("overfit mode does not accept --require-overfit-go")
        prerequisite = None

    model, receiver_tokenizer, sharer_tokenizer = build_model(
        receiver_path=str(manifest["receiver"]),
        sharer_path=str(manifest["sharer"]),
        layer_mapping=manifest["layer_mapping"],
        device_name=args.device,
    )
    receiver_prompt = verify_receiver_prompt(receiver_tokenizer, manifest)
    collator = DraftKVReconstructionCollator(
        receiver_tokenizer, sharer_tokenizer
    )
    if args.mode == "overfit":
        allowed = {str(value) for value in manifest["overfit_ids"]}
        train_records = [row for row in records if str(row["example_id"]) in allowed]
        validation_records = train_records
        validation_split = "train"
    else:
        train_records = [row for row in records if row["split"] == "train"]
        validation_records = [row for row in records if row["split"] == "gate_val"]
        validation_split = "gate_val"
    train_dataset = _make_dataset(
        train_records,
        receiver_tokenizer,
        sharer_tokenizer,
        manifest,
        split="train",
    )
    validation_dataset = _make_dataset(
        validation_records,
        receiver_tokenizer,
        sharer_tokenizer,
        manifest,
        split=validation_split,
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.microbatch),
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    first_batch = next(iter(train_loader))
    parity = parity_checks(model, first_batch)
    if parity["native_vs_split_max_abs"] > 1e-5:
        raise RuntimeError(f"native/split parity failed: {parity}")
    if parity["base_vs_zero_gate_max_abs"] != 0.0:
        raise RuntimeError(f"zero-gate parity failed: {parity}")

    model.set_stage("communication")
    projector_parameters = list(model.projection.parameters())
    gate_parameters = list(model.consumer.gate_parameters)
    trainable = list(model.communication_parameters)
    trainable_names = model.trainable_parameter_names()
    if not trainable or any(
        not (name.startswith("projection.") or "gate_logits" in name)
        for name in trainable_names
    ):
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names}")
    expected_gate_count = len(manifest["layer_mapping"]) * int(
        model.receiver.config.num_key_value_heads
    )
    if count_parameters(gate_parameters) != expected_gate_count:
        raise RuntimeError("unexpected gate parameter count")
    optimizer = torch.optim.AdamW(
        [
            {"params": projector_parameters, "lr": float(args.projector_lr)},
            {"params": gate_parameters, "lr": float(args.gate_lr)},
        ],
        weight_decay=0.0,
    )

    initial_validation = validation_nll(model, validation_loader)
    history: list[Dict[str, Any]] = [
        {"optimizer_update": 0, "validation": initial_validation}
    ]
    print("validation", json.dumps(history[-1]), flush=True)
    model.set_stage("communication")
    optimizer.zero_grad(set_to_none=True)
    update = 0
    microstep = 0
    epoch = 0
    best_gap = float("-inf")
    best_update = -1
    start_time = time.time()
    while update < int(args.max_updates):
        epoch += 1
        for batch in train_loader:
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
                return_diagnostics=(microstep == 0),
                use_cache=False,
            )
            stats = token_reconstruction_stats(logits, batch["labels"])
            loss = stats["sum"].sum() / stats["count"].sum().clamp_min(1)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite reconstruction loss at update {update}")
            (loss / float(args.grad_accum)).backward()
            microstep += 1
            if microstep % int(args.grad_accum):
                continue
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
            ):
                raise RuntimeError(
                    f"non-finite communication gradient at update {update}"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if update == 1 or update % int(args.log_every) == 0:
                row = {
                    "epoch": epoch,
                    "optimizer_update": update,
                    "loss": float(loss.detach()),
                    "token_accuracy": float(
                        stats["correct"].sum() / stats["count"].sum().clamp_min(1)
                    ),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "diagnostic_layers": sorted(diagnostics),
                }
                history.append(row)
                print("train", json.dumps(row), flush=True)
            should_evaluate = (
                update % int(args.eval_every) == 0
                or update == int(args.max_updates)
            )
            if should_evaluate:
                validation = validation_nll(model, validation_loader)
                row = {"optimizer_update": update, "validation": validation}
                history.append(row)
                print("validation", json.dumps(row), flush=True)
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    records_path=records_path,
                    args=args,
                    update=update,
                    validation=validation,
                    prerequisite=prerequisite,
                )
                torch.save(payload, output / "last.pt")
                gap = float(validation["zero_minus_matched_nll"])
                if gap > best_gap:
                    best_gap = gap
                    best_update = update
                    torch.save(payload, output / "best.pt")
                model.set_stage("communication")
            if update >= int(args.max_updates):
                break

    model.set_stage("eval")
    gates = {
        layer: torch.tanh(module.gate_logits).float().cpu().tolist()
        for layer, module in model.consumer.external.items()
    }
    result = {
        "protocol": PROTOCOL,
        "reconstruction_mode": str(manifest.get("reconstruction_mode", "base")),
        "mode": args.mode,
        "best_checkpoint": str((output / "best.pt").resolve()),
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "last_checkpoint": str((output / "last.pt").resolve()),
        "last_checkpoint_sha256": sha256_file(output / "last.pt"),
        "best_update": int(best_update),
        "best_zero_minus_matched_nll": float(best_gap),
        "parity": parity,
        "receiver_prompt": receiver_prompt,
        "prerequisite": prerequisite,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "optimizer_updates": update,
        "projector_parameter_count": count_parameters(projector_parameters),
        "gate_parameter_count": count_parameters(gate_parameters),
        "total_trainable_parameter_count": count_parameters(trainable),
        "trainable_parameter_names": list(trainable_names),
        "gates_tanh_last": gates,
        "runtime_seconds": time.time() - start_time,
    }
    write_json(output / "train_history.json", history)
    write_json(output / "train_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
